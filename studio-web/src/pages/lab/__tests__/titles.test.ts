// No native `title` tooltips in the Scenario Lab (phase 4): an explanation is a Tooltip, an InfoTip
// (ⓘ) or a Term, which open on hover, keyboard focus and tap, and read the same words for everyone.
// The guard reads the sources: a `title=` on a DOM element (a lower-case JSX tag) or on a component
// that passes it to the DOM as a native title (Button, Pill, Badge, StatusChip, Link, Kpi) fails.
// Component props that are headings (a chart's, a table's, a dialog's or the page frame's `title`)
// are fine. Climate.tsx has its own owner and test; Scenarios and Budget (run pages whose canonical
// home is the Lab) are covered too.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const HERE = dirname(fileURLToPath(import.meta.url));
const LAB = resolve(HERE, "..");
const RUNHUB = resolve(LAB, "../runhub");

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) {
      if (!/^__(tests|fixtures)__$/.test(name)) walk(p, out);
    } else if (p.endsWith(".tsx")) out.push(p);
  }
  return out;
}

/** Components that render their `title` prop as a native title attribute. */
const NATIVE = /^(?:[a-z][\w-]*|Button|Pill|Badge|StatusChip|Link|Kpi)$/;

/** Each `title=` whose element (the last JSX tag opened before it) shows it as a native tooltip: "file:line <tag>". */
function nativeTitles(file: string, src: string): string[] {
  const out: string[] = [];
  for (const m of src.matchAll(/\btitle=[{"]/g)) {
    const before = src.slice(0, m.index);
    const tag = [...before.matchAll(/<([A-Za-z][\w.]*)/g)].at(-1)?.[1];
    if (tag && NATIVE.test(tag)) out.push(`${file}:${before.split("\n").length} <${tag}>`);
  }
  return out;
}

describe("native titles", () => {
  it("finds a native title on a DOM element or a Button, not on a chart's or a dialog's heading", () => {
    expect(nativeTitles("x.tsx", `<span className="a" title={tip}>x</span>`)).toEqual(["x.tsx:1 <span>"]);
    expect(nativeTitles("x.tsx", `<Button\n  size="small"\n  title="Why"\n>Go</Button>`)).toEqual(["x.tsx:3 <Button>"]);
    expect(nativeTitles("x.tsx", `<LineBand title="Cooling by amount" />\n<Dialog open title="Delete?" />`)).toEqual([]);
  });

  it("the Lab and its run pages (Scenarios, Budget) explain things in tooltips, never a native title", () => {
    const files = [...walk(LAB).filter((f) => relative(LAB, f) !== "Climate.tsx"), join(RUNHUB, "Scenarios.tsx"), join(RUNHUB, "Budget.tsx")];
    const found = files.flatMap((f) => nativeTitles(relative(resolve(LAB, "../.."), f), readFileSync(f, "utf8")));
    expect(files.length).toBeGreaterThan(20);
    expect(found).toEqual([]);
  });
});
