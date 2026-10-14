// Phase 4 (plain language): no native `title="…"` tooltips anywhere in the pages or the shared
// components. A native title shows only on mouse hover after a long delay, never on keyboard focus
// or a tap, and is read inconsistently by screen readers; an explanation is a Tooltip, an InfoTip
// (ⓘ) or a Term instead (components/ui/Tooltip.tsx, useTipOn.ts).
//
// The guard reads the sources (test files excluded): a `title=` attribute, a string or an
// expression, on an HTML element (a lower-case JSX tag) fails, listed as "file:line <tag>".
// Component props named `title` (a chart's, a card's, a dialog's or a page section's heading) are
// fine, and so are:
// - <iframe title> — the frame's accessible name, which WCAG requires (not a tooltip);
// - the Markdown renderer, which writes a document's own link and image titles into HTML strings.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(HERE, "..");

/** Every .ts/.tsx source under `dir`, skipping tests and test fixtures. */
function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) {
      if (!/^__(tests|fixtures)__$/.test(name)) walk(p, out);
    } else if (/\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

/** HTML elements whose `title` is not a tooltip. */
const TITLE_IS_NAME = new Set(["iframe"]);

/** Sources allowed a native title (see the file comment). */
const ALLOWED = new Set(["components/ui/Markdown.tsx"]);

/** Each `title=` on an HTML element (the last JSX tag opened before it is lower case): "file:line <tag>". */
function nativeTitles(file: string, src: string): string[] {
  const out: string[] = [];
  for (const m of src.matchAll(/\btitle=(?:"[^"]|\{)/g)) {
    const before = src.slice(0, m.index);
    const tag = [...before.matchAll(/<([A-Za-z][\w.-]*)/g)].at(-1)?.[1];
    if (tag && /^[a-z]/.test(tag) && !TITLE_IS_NAME.has(tag)) out.push(`${file}:${before.split("\n").length} <${tag}>`);
  }
  return out;
}

describe("no native titles", () => {
  it("finds a title on an HTML element, not on a component's heading prop or an iframe's name", () => {
    expect(nativeTitles("x.tsx", `<span className="a" title="Why">x</span>`)).toEqual(["x.tsx:1 <span>"]);
    expect(nativeTitles("x.tsx", `<div>\n  <td\n    key={k}\n    title={label}\n  />\n</div>`)).toEqual(["x.tsx:4 <td>"]);
    expect(nativeTitles("x.tsx", `<Section title="Cooling by amount" />\n<ChartFrame title={t} />\n<Card title="Inputs" />`)).toEqual([]);
    expect(nativeTitles("x.tsx", `<iframe title="Report preview" src={u} />`)).toEqual([]);
    expect(nativeTitles("x.tsx", `<span title="">x</span>`)).toEqual([]); // an empty title shows nothing
  });

  it("src/pages and src/components explain things with Tooltip, InfoTip or Term", () => {
    const files = [...walk(join(SRC, "pages")), ...walk(join(SRC, "components"))];
    expect(files.length).toBeGreaterThan(150);
    const found = files.filter((f) => !ALLOWED.has(relative(SRC, f))).flatMap((f) => nativeTitles(relative(SRC, f), readFileSync(f, "utf8")));
    expect(found).toEqual([]);
  });
});
