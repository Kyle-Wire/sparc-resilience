// Phase 4 (plain language): the shared UI components, charts and map explain things in tooltips
// (hover, keyboard focus, tap; never a native title), and use the glossary's plain words — a grid
// cell is a "place", a stage is its plain name with the code in the tooltip.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { act } from "react";
import { describe, expect, it } from "vitest";
import { LineBand } from "../charts";
import { term } from "../content/glossary";
import { Badge, DemoBadge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Chips } from "../components/ui/Chips";
import { IconButton } from "../components/ui/IconButton";
import { stagePlain, stageTermId } from "../components/ui/JobStrip";
import { Kpi } from "../components/ui/Kpi";
import { Pill } from "../components/ui/Pill";
import { Seg } from "../components/ui/Seg";
import { StatusChip } from "../components/ui/StatusChip";
import { Inspector, type CellInfo } from "../map/Inspector";
import { pickSelection, placesText, rectSelection } from "../map/tools";
import { grid3 } from "./grid";
import { byText, click, key, render } from "./render";

const tooltip = () => document.querySelector<HTMLElement>('[role="tooltip"]:not([hidden])');
const focus = (el: Element) => act(() => (el as HTMLElement).focus());
const blur = (el: Element) => act(() => (el as HTMLElement).blur());

describe("chips, badges and pills explain themselves in a tooltip", () => {
  it("a StatusChip with a tip is focusable and shows it on focus; Escape closes it; no native title", () => {
    const { container } = render(<StatusChip status="skipped" tip="No budget was set, so this step did not run." />);
    const chip = container.querySelector<HTMLElement>(".status")!;
    expect(chip.hasAttribute("title")).toBe(false);
    expect(chip.tabIndex).toBe(0);
    expect(tooltip()).toBeNull();
    focus(chip);
    expect(tooltip()!.textContent).toBe("No budget was set, so this step did not run.");
    expect(chip.getAttribute("aria-describedby")).toBe(tooltip()!.id);
    key(chip, "Escape");
    expect(tooltip()).toBeNull();
    // without a tip: a plain, unfocusable chip
    const plain = render(<StatusChip status="done" />).container.querySelector(".status")!;
    expect(plain.hasAttribute("tabindex")).toBe(false);
  });

  it("the deprecated title prop of Badge, Pill and StatusChip is a tooltip too", () => {
    const { container } = render(
      <>
        <Badge title="Built from a preview">preview</Badge>
        <Pill title="Not recorded for one of the runs">unknown</Pill>
        <StatusChip status="done" title="Step check" />
      </>,
    );
    expect(container.querySelector("[title]")).toBeNull();
    focus(container.querySelector(".pill")!);
    expect(tooltip()!.textContent).toBe("Not recorded for one of the runs");
  });

  it("DEMO says what demo data is, from the glossary", () => {
    const { container } = render(<DemoBadge />);
    focus(container.querySelector(".badge")!);
    expect(tooltip()!.textContent).toContain(term("demo").tip);
    expect(tooltip()!.textContent).toContain(`Technical: ${term("demo").technical}`);
  });
});

describe("Kpi", () => {
  it("term= adds an ⓘ after the label that explains the glossary term", () => {
    const { container } = render(<Kpi label="Accuracy score" value="0.71" term="r2" />);
    const info = container.querySelector<HTMLButtonElement>(".kpi .label button.infotip")!;
    expect(info.getAttribute("data-term")).toBe("r2");
    expect(info.getAttribute("aria-label")).toBe("What is Accuracy score?");
    focus(info);
    expect(tooltip()!.textContent).toContain(term("r2").tip);
  });

  it("tip= (or the old title=) is a one-off ⓘ, never a native title", () => {
    const { container } = render(
      <>
        <Kpi label="Residents" value="12,400" tip="People living in the places you changed." />
        <Kpi label="Cost" value="$1.2M" title="Estimated from the cost per unit." />
      </>,
    );
    expect(container.querySelector("[title]")).toBeNull();
    const [a, b] = [...container.querySelectorAll<HTMLButtonElement>("button.infotip")];
    expect(a.getAttribute("aria-label")).toBe("What is Residents?");
    focus(b);
    expect(tooltip()!.textContent).toBe("Estimated from the cost per unit.");
  });
});

describe("buttons", () => {
  it("IconButton shows its label in a tooltip on focus, keeps its own handlers, and has no native title", () => {
    const seen: string[] = [];
    const { container } = render(<IconButton icon="x" label="Close" onFocus={() => seen.push("focus")} onClick={() => seen.push("click")} />);
    const btn = container.querySelector("button")!;
    expect(btn.hasAttribute("title")).toBe(false);
    expect(btn.getAttribute("aria-label")).toBe("Close");
    focus(btn);
    expect(tooltip()!.textContent).toBe("Close");
    expect(btn.hasAttribute("aria-describedby")).toBe(false); // the tip repeats the name
    click(btn);
    expect(seen).toEqual(["focus", "click"]);
    expect(tooltip()).not.toBeNull(); // a click acts; it does not toggle the tooltip
    blur(btn);
    expect(tooltip()).toBeNull();
    // a longer tip describes the button
    const t = render(<IconButton icon="image" label="Export" title="Export the map as a PNG with its legend" />).container.querySelector("button")!;
    focus(t);
    expect(t.getAttribute("aria-describedby")).toBe(tooltip()!.id);
  });

  it("Button tip= explains a disabled button on hover, and describes it for screen readers", () => {
    let n = 0;
    const { container, rerender } = render(
      <Button disabled tip="Add a change first." onClick={() => n++}>
        Run full model
      </Button>,
    );
    const btn = byText(container, "button", "Run full model")!;
    expect(btn.hasAttribute("title")).toBe(false);
    expect(document.getElementById(btn.getAttribute("aria-describedby")!)!.textContent).toBe("Add a change first.");
    expect(btn.parentElement!.classList.contains("btn-tip")).toBe(true); // the hover target around the disabled button
    // enabled again: the same element (focus is kept), still described, and it acts on click
    rerender(
      <Button tip="Runs your changes through the full model." onClick={() => n++}>
        Run full model
      </Button>,
    );
    expect(byText(container, "button", "Run full model")).toBe(btn);
    focus(btn);
    expect(tooltip()!.textContent).toBe("Runs your changes through the full model.");
    click(btn);
    expect(n).toBe(1);
  });

  it("Seg and Chips options carry tips without native titles; clicks still choose", () => {
    let v = "a";
    const toggled: string[] = [];
    const { container } = render(
      <>
        <Seg label="Mode" value={v} onChange={(x) => (v = x)} options={[{ value: "a", label: "A", tip: "First" }, { value: "b", label: "B" }, { value: "c", label: "C", disabled: true, tip: "Needs the same grid" }]} />
        <Chips label="Futures" items={[{ value: "x", label: "X", tip: "Moderate emissions" }]} selected={[]} onToggle={(x) => toggled.push(x)} />
      </>,
    );
    expect(container.querySelector("[title]")).toBeNull();
    const a = byText(container, ".seg button", "A")!;
    focus(a);
    expect(tooltip()!.textContent).toBe("First");
    blur(a);
    click(byText(container, ".seg button", "B"));
    expect(v).toBe("b");
    const c = byText(container, ".seg button", "C")!;
    expect(c.hasAttribute("disabled")).toBe(true);
    expect(document.getElementById(c.getAttribute("aria-describedby")!)!.textContent).toBe("Needs the same grid");
    const chip = byText(container, ".chips button", "X")!;
    focus(chip);
    expect(tooltip()!.textContent).toBe("Moderate emissions");
    click(chip);
    expect(toggled).toEqual(["x"]);
  });
});

describe("charts and the job strip", () => {
  it("a chart's term= puts an ⓘ beside its title (not inside the heading)", () => {
    const { container } = render(<LineBand title="Cooling by amount" term="dose_response" series={[{ id: "s", label: "s", x: [0, 1], y: [0, -0.2] }]} xLabel="Amount" yLabel="Cooling" />);
    const h3 = container.querySelector("figure.chart-frame h3")!;
    expect(h3.textContent).toBe("Cooling by amount");
    expect(h3.querySelector("button")).toBeNull();
    expect(container.querySelector('.chart-title button.infotip[data-term="dose_response"]')).not.toBeNull();
  });

  it("names a pipeline stage in plain words from the glossary", () => {
    expect(stageTermId("S2_S3")).toBe("stage_s2_s3");
    expect(stageTermId("cv_curve")).toBe("stage_cv_curve");
    expect(stageTermId("mystery")).toBeNull();
    expect(stagePlain("S4")).toBe(term("stage_s4").label);
    expect(stagePlain("mystery")).toBe("mystery");
  });
});

describe("map words", () => {
  it("selection labels count places, with the tool's own name", () => {
    expect(placesText(1)).toBe("1 place");
    expect(placesText(1234)).toBe("1,234 places");
    const g = grid3();
    expect(rectSelection(g, { px: 0, py: 0 }, { px: 3, py: 3 }).label).toBe("Box · 7 places");
    expect(pickSelection(g, { by: "zone" }, [1]).label).toBe("Neighbourhood 1 · 3 places");
    expect(pickSelection(g, { by: "zone" }, [1, 2]).label).toBe("2 neighbourhoods · 7 places");
  });

  it("the pinned-place card says place, keeps the layer key in a tooltip and names the lever's curve plainly", () => {
    const cell: CellInfo = {
      index: 3, id: 1001, lon: -71.4, lat: 41.8, zone: 2,
      values: { Pct_Canopy: 23.5 },
      curves: { canopy: { model: "saturating", A: 2, ds: 10, inflection: null, d90: 23, dmax: 40, dose: [0, 10, 20], benefit: [0, -0.5, -0.8] } },
      scenarios: {},
    };
    const layers = [{ key: "Pct_Canopy", group: "inputs", label: "Tree canopy", unit: "pct", scale: "seq", center: null, decimals: 1, mult: 1, zero_blank: false, labels: null, desc: "Share of the place under tree canopy.", sign_note: null, source: null, dtype: "float32", stats: null }] as never;
    const { container } = render(<Inspector cell={cell} layers={layers} targetUnit="degF" onClose={() => {}} />);
    const card = container.querySelector('[aria-label="Pinned place"]')!;
    expect(card.querySelector("h3")!.textContent).toBe("Place 1001");
    expect(card.querySelector('button[aria-label="Unpin this place"]')).not.toBeNull();
    expect(card.querySelector("[title]")).toBeNull();
    const dt = byText(card, "dt", "Tree canopy")!;
    act(() => {
      dt.querySelector(".tip-anchor")!.dispatchEvent(new MouseEvent("mouseover", { bubbles: true, relatedTarget: null }));
    });
    expect(card.querySelector('[role="tooltip"]')!.textContent).toContain("Technical: Pct_Canopy");
    expect(card.querySelector('svg.chart[aria-label="Trees & canopy: cooling by amount here"]')).not.toBeNull();
    expect(card.textContent).toContain("90% of the cooling");
  });
});

// ---------------------------------------------------------------- guard

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(HERE, "..");

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) {
      if (!/^__(tests|fixtures)__$/.test(name)) walk(p, out);
    } else if (p.endsWith(".tsx")) out.push(p);
  }
  return out;
}

/** Components that still pass a `title` prop to the DOM as a native title. */
const NATIVE = /^(?:[a-z][\w-]*|Button|Link)$/;

/** Each `title=` on a DOM element or a native-title component: "file:line <tag>". */
function nativeTitles(file: string, src: string): string[] {
  const out: string[] = [];
  for (const m of src.matchAll(/\btitle=[{"]/g)) {
    const before = src.slice(0, m.index);
    const tag = [...before.matchAll(/<([A-Za-z][\w.]*)/g)].at(-1)?.[1];
    if (tag && NATIVE.test(tag)) out.push(`${file}:${before.split("\n").length} <${tag}>`);
  }
  return out;
}

/** Files allowed a native title: the Markdown renderer (a link's or image's own title from the document). */
const ALLOWED = new Set(["components/ui/Markdown.tsx"]);

describe("no native titles in the shared UI, the charts and the map", () => {
  it("explains things with Tooltip, InfoTip or Term instead", () => {
    const files = ["components/ui", "charts", "map"].flatMap((d) => walk(join(SRC, d)));
    expect(files.length).toBeGreaterThan(40);
    const found = files.filter((f) => !ALLOWED.has(relative(SRC, f))).flatMap((f) => nativeTitles(relative(SRC, f), readFileSync(f, "utf8")));
    expect(found).toEqual([]);
  });
});
