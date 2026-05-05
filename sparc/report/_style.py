"""
Shared brand styling for SPARC Stage 5 report surfaces.

Single source of truth for the warm-paper palette, SPARC color ramp,
typography, and component classes used by every HTML report emitter
(:mod:`sparc.report.__init__`, :mod:`sparc.report.standalone_html`,
:mod:`sparc.report.audience`) and by the matplotlib figure renderer
(:mod:`sparc.report.figures`).

The CSS mirrors the marketing site (`tokens.css` + `globals.css`) but is
inlined so reports remain self-contained, print-safe, and offline-friendly.
Web fonts are loaded from Google Fonts when available and fall back to
``Georgia`` / system-mono otherwise.
"""

from __future__ import annotations


# 9-stop SPARC color ramp (matches `tokens.css` --sparc-*).
RAMP_HEX: list[str] = [
    "#602468",  # purple
    "#9e337d",  # magenta
    "#e94d9b",  # pink
    "#e94461",  # red
    "#e73c25",  # crimson
    "#e76c25",  # orange
    "#e79024",  # amber
    "#f0b632",  # gold
    "#fbdd46",  # yellow
]

RAMP_RGB: list[tuple[int, int, int]] = [
    (96, 36, 104),
    (158, 51, 125),
    (233, 77, 155),
    (233, 68, 97),
    (231, 60, 37),
    (231, 108, 37),
    (231, 144, 36),
    (240, 182, 50),
    (251, 221, 70),
]


# Inlined design tokens + components. Intentionally trimmed to what
# Stage 5 reports actually use; keep selectors stable so the existing
# Jinja template / audience wrapper / standalone bundle don't break.
BRAND_CSS: str = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --sparc-purple:  #602468;
  --sparc-magenta: #9e337d;
  --sparc-pink:    #e94d9b;
  --sparc-red:     #e94461;
  --sparc-crimson: #e73c25;
  --sparc-orange:  #e76c25;
  --sparc-amber:   #e79024;
  --sparc-gold:    #f0b632;
  --sparc-yellow:  #fbdd46;

  --purple:  var(--sparc-purple);
  --magenta: var(--sparc-magenta);
  --pink:    var(--sparc-pink);
  --red:     var(--sparc-red);
  --crimson: var(--sparc-crimson);
  --orange:  var(--sparc-orange);
  --amber:   var(--sparc-amber);
  --gold:    var(--sparc-gold);
  --yellow:  var(--sparc-yellow);

  --ink:     #1a1416;
  --ink-2:   #2b2327;
  --paper:   #f7f4ee;
  --paper-2: #ede8dd;
  --line:    #c9c2b3;
  --muted:   #6e6358;
  --white:   #ffffff;

  --bg:           var(--paper);
  --bg-elevated:  var(--white);
  --bg-sunken:    var(--paper-2);
  --fg:           var(--ink);
  --fg-2:         var(--ink-2);
  --fg-muted:     var(--muted);
  --border:       var(--line);
  --accent:       var(--crimson);
  --accent-2:     var(--purple);

  --status-pass: #166534;
  --status-fail: #991b1b;
  --status-warn: var(--amber);

  --font-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
  --font-serif-fallback: Georgia, "Iowan Old Style", "Times New Roman", serif;

  --grad-ramp: linear-gradient(90deg,
    var(--purple) 0%, var(--magenta) 14%, var(--pink) 28%, var(--red) 42%,
    var(--crimson) 56%, var(--orange) 70%, var(--amber) 82%, var(--gold) 92%, var(--yellow) 100%);
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--fg);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}

.wrap { max-width: 1100px; margin: 0 auto; padding: 0 32px; }
.wrap-narrow { max-width: 880px; margin: 0 auto; padding: 0 32px; }

h1, h2, h3, h4, h5, h6 { margin: 0; color: var(--fg); letter-spacing: -0.01em; }
h1 { font-size: 36px; font-weight: 800; line-height: 1.05; letter-spacing: -0.02em; }
h2 { font-size: 22px; font-weight: 700; line-height: 1.2; margin: 28px 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--line); color: var(--ink); }
h3 { font-size: 14px; font-weight: 700; line-height: 1.25; margin: 20px 0 8px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-2); font-family: var(--font-mono); }
h4 { font-size: 14px; font-weight: 600; margin: 0 0 6px; }
p  { margin: 0 0 12px; }
a  { color: var(--ink); text-decoration: underline; text-underline-offset: 3px; text-decoration-color: var(--line); }
a:hover { text-decoration-color: var(--crimson); }
em { color: var(--muted); font-style: normal; }

code, pre, kbd, samp, .mono { font-family: var(--font-mono); font-size: 12px; color: var(--ink-2); }
code { background: rgba(96, 36, 104, 0.08); padding: 1px 6px; border-radius: 3px; color: var(--purple); }
pre  { background: var(--white); border: 1px solid var(--line); border-radius: 6px; padding: 12px; overflow: auto; }

blockquote {
  margin: 0 0 16px;
  padding: 10px 14px;
  background: var(--white);
  border-left: 3px solid var(--crimson);
  color: var(--ink-2);
  border-radius: 0 4px 4px 0;
}

/* Page hero (matches site .page-hero). */
.page-hero {
  padding: 40px 0 24px;
  border-bottom: 1px solid var(--line);
  background: var(--paper);
}
.page-hero .kicker { display: inline-block; margin-bottom: 10px; }
.page-hero h1 { margin: 0 0 10px; max-width: 880px; }
.page-hero .subtitle { color: var(--muted); font-size: 14px; }
.page-hero .meta { color: var(--muted); font-size: 12px; margin-top: 8px; }

/* SPARC ramp strip (purely decorative; degrades to solid in print). */
.ramp-strip {
  height: 4px; width: 100%;
  background: var(--grad-ramp);
}

/* Kicker (mono uppercase label) + pill / tag. */
.kicker {
  font-family: var(--font-mono); font-size: 10px;
  text-transform: uppercase; letter-spacing: 0.14em; color: var(--muted);
}
.tag, .pill {
  display: inline-block;
  font-family: var(--font-mono); font-size: 10px;
  text-transform: uppercase; letter-spacing: 0.1em;
  padding: 2px 8px; border-radius: 9999px;
  background: rgba(96, 36, 104, 0.10); color: var(--purple);
}
.tag.crimson { background: rgba(231, 60, 37, 0.10); color: var(--crimson); }
.tag.amber   { background: rgba(231, 144, 36, 0.18); color: #7d5e0e; }

/* Cards. */
.card {
  background: var(--bg-elevated);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
  margin-bottom: 16px;
}

/* Section content wrapper. */
main { padding: 32px 0 64px; }

/* Tables. */
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0 18px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--line); }
th {
  background: var(--bg-sunken);
  color: var(--muted);
  font-family: var(--font-mono); font-size: 10px;
  font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
}
tr:nth-child(even) td { background: rgba(0, 0, 0, 0.015); }

/* Badges (status indicators). */
.badge {
  display: inline-block;
  font-family: var(--font-mono); font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.08em;
  padding: 2px 8px; border-radius: 9999px;
  background: rgba(0, 0, 0, 0.05); color: var(--ink-2);
}
.badge-pass { background: #dcfce7; color: var(--status-pass); }
.badge-fail { background: #fee2e2; color: var(--status-fail); }
.badge-info { background: rgba(96, 36, 104, 0.10); color: var(--purple); }

/* Stat grids (Overview metric cards). */
.stat-grid, .grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin: 12px 0 18px;
}
.stat-card, .stat {
  background: var(--bg-elevated);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 14px;
}
.stat-card .value, .stat .v {
  display: block;
  font-size: 22px; font-weight: 700; color: var(--ink);
  letter-spacing: -0.01em;
}
.stat-card .label, .stat .l {
  display: block; margin-top: 4px;
  font-family: var(--font-mono); font-size: 10px;
  text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
}

/* Treatment-effect cards (sign-coded, semantic preserved). */
.treatment-card {
  border: 1px solid var(--line);
  border-left-width: 4px;
  border-radius: 6px;
  padding: 14px;
  margin: 10px 0;
  background: var(--bg-elevated);
  page-break-inside: avoid;
}
.treatment-card.negative { border-left-color: var(--purple);  background: rgba(96, 36, 104, 0.04); }
.treatment-card.positive { border-left-color: var(--crimson); background: rgba(231, 60, 37, 0.04); }

/* Plot gallery. */
.plot-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 14px;
  margin: 12px 0 18px;
}
.plot-img {
  max-width: 100%;
  display: block;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--bg-elevated);
}
.plot-caption {
  font-family: var(--font-mono); font-size: 10px;
  text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); text-align: center; margin-top: 4px;
}

/* Tab nav (used by standalone bundle). */
.nav {
  position: sticky; top: 0; z-index: 10;
  display: flex; gap: 4px;
  padding: 0 32px;
  background: color-mix(in srgb, var(--paper) 92%, transparent);
  border-bottom: 1px solid var(--line);
}
.nav button {
  background: none; border: 0;
  padding: 14px 16px;
  font: inherit; font-size: 13px; font-weight: 500;
  color: var(--muted); cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: color .15s, border-color .15s;
}
.nav button:hover { color: var(--ink); }
.nav button.active { color: var(--ink); border-bottom-color: var(--accent); }

/* Tab panel show/hide (preserved behavior). */
.section { display: none; }
.section.active { display: block; }

/* Chat messages. */
.chat-msg { padding: 10px 14px; border-radius: 6px; margin-bottom: 8px; background: var(--bg-elevated); border: 1px solid var(--line); }
.chat-msg.user      { background: rgba(96, 36, 104, 0.05); border-left: 3px solid var(--purple); }
.chat-msg.assistant { background: rgba(231, 60, 37, 0.04); border-left: 3px solid var(--crimson); }

/* Helpers. */
.muted, .subtitle { color: var(--muted); }
.footer {
  margin-top: 40px;
  padding: 16px 0;
  border-top: 1px solid var(--line);
  font-family: var(--font-mono); font-size: 10px;
  text-transform: uppercase; letter-spacing: 0.12em;
  color: var(--muted); text-align: center;
}

/* Audience-specific override: keep public-facing report softer. */
body.public h2 { font-size: 22px; color: var(--crimson); border: 0; padding: 0; }
"""


# Print-safe overrides. Loaded after BRAND_CSS so it always wins.
BRAND_CSS_PRINT: str = """
@media print {
  @page { size: letter; margin: 2cm; }
  body { background: var(--white); font-size: 11pt; }
  .nav, nav { display: none !important; }
  .section { display: block !important; }
  .ramp-strip { background: var(--purple) !important; }
  .card, .stat-card, .stat, .treatment-card { box-shadow: none; }
  .treatment-card, table, .plot-gallery > * { page-break-inside: avoid; }
  h2 { page-break-after: avoid; }
  a { color: inherit; text-decoration: none; }
}
"""


__all__ = ["BRAND_CSS", "BRAND_CSS_PRINT", "RAMP_HEX", "RAMP_RGB"]
