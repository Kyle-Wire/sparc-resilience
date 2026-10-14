// Plain words and tooltips (phase 4) in the run hub's shared blocks and Model details: status chips
// explain themselves in a tooltip (TipChip, no native title), KPI tiles and block titles take an ⓘ
// beside them (never inside the heading), the caveats read "Things to keep in mind", the view's
// availability is in plain words, and Model details keeps its technical names with an ⓘ or a
// glossary <Term> for each (section titles, intros, Data & checks tiles and blocks, Provenance,
// Files, the run record).
import { act, type ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import type { ViewKpi } from "../../../api/runs";
import { term } from "../../../content/glossary";
import { navigate } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { flush, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import { viewFixture } from "../__fixtures__/views";
import { AVAILABILITY_WORDS, Caveats, KpiTiles, OlderCode, TipChip, VerdictPill } from "../common";
import DataQa from "../DataQa";
import Details, { DETAIL_SECTION_TIPS } from "../Details";
import { RunRecord } from "../Overview";
import Provenance from "../Provenance";
import { runDetail, runSummary } from "./helpers";

let seq = 0;
const nextRid = (p: string) => `r_plain_${p}${++seq}`;

/** Focus an element and return a copy of the tooltip it opens (InfoTip, Term, TipChip), then close it. */
function tipOf(el: Element): HTMLElement {
  act(() => (el as HTMLElement).focus());
  const tip = document.querySelector<HTMLElement>('[role="tooltip"]:not([hidden])');
  expect(tip, "a tooltip opens on focus").not.toBeNull();
  const copy = tip!.cloneNode(true) as HTMLElement;
  act(() => (el as HTMLElement).blur());
  return copy;
}

async function mount(path: string, ui: ReactElement, routes: Record<string, MockHandler>, ready: (root: HTMLElement) => unknown) {
  const m = mockFetch(routes);
  navigate(path, { replace: true });
  const r = render(ui);
  await waitFor(() => ready(r.container), 10000, path);
  await flush(4);
  return { ...r, m };
}

beforeEach(() => useJobs.setState({ jobs: {} }));
afterEach(() => clearResources());

describe("shared blocks", () => {
  it("KpiTiles puts an ⓘ after the labels it names; the label text stays the KPI's", () => {
    const kpis: ViewKpi[] = [
      { id: "r2", label: "Held-out R²", value: 0.81, decimals: 2 } as ViewKpi,
      { id: "cells", label: "Cells treated", value: 12, format: "int" } as ViewKpi,
    ];
    const { container } = render(<KpiTiles kpis={kpis} tips={{ r2: { term: "r2" } }} />);
    const [r2, cells] = [...container.querySelectorAll(".kpi")];
    expect(r2.querySelector(".label")!.textContent).toBe("Held-out R²");
    expect(r2.querySelector("button.infotip")!.getAttribute("aria-label")).toBe(`What is ${term("r2").label}?`);
    expect(tipOf(r2.querySelector("button.infotip")!).querySelector(".tip-tech code")!.textContent).toBe(term("r2").technical);
    expect(cells.querySelector("button")).toBeNull();
  });

  it("Caveats reads 'Things to keep in mind (n)', as on the Summary", () => {
    const { container } = render(<Caveats caveats={["A", "B"]} open />);
    expect(container.querySelector("summary")!.textContent).toBe("Things to keep in mind (2)");
  });

  it("a verdict's reasons are in its tooltip, not a native title; the pill's words are unchanged", () => {
    const { container } = render(<VerdictPill verdict={{ verdict: "direction", label: "Direction only", reasons: ["range crosses zero"], qualifiers: ["extrapolated"] }} />);
    expect(container.querySelector("[title]")).toBeNull();
    expect(container.textContent).toBe("Direction only *");
    expect(tipOf(container.querySelector("[data-tip-chip]")!).textContent).toBe("range crosses zero; note: extrapolated");
    // nothing to explain: the bare pill
    const bare = render(<VerdictPill verdict={{ verdict: "robust", label: "Robust", reasons: [], qualifiers: [] }} />);
    expect(bare.container.querySelector("[data-tip-chip]")).toBeNull();
  });

  it("TipChip adds no text to the page and is reachable from the keyboard", () => {
    const { container } = render(
      <TipChip tip="What this means">
        <span>chip</span>
      </TipChip>,
    );
    const chip = container.querySelector<HTMLElement>("[data-tip-chip]")!;
    expect(container.textContent).toBe("chip");
    expect(chip.tabIndex).toBe(0);
    expect(tipOf(chip).textContent).toBe("What this means");
  });

  it("an older-code placeholder keeps its words and explains them in an ⓘ", () => {
    const { container } = render(<OlderCode title="Forcing" />);
    expect(container.textContent).toBe("Forcing: not in this run (older code)");
    expect(tipOf(container.querySelector("button.infotip")!).textContent).toMatch(/earlier version of SPARC/);
  });
});

describe("ViewPage header", () => {
  it("shows the view's availability in plain words and explains DEMO in a tooltip, with no native titles", async () => {
    const rid = nextRid("vp");
    const data = { ...viewFixture("data"), availability: "stale" as const };
    const { container, m } = await mount(
      `/r/${rid}/details/data`,
      <DataQa />,
      { [`GET /api/runs/${rid}`]: { body: runDetail(rid) }, [`GET /api/runs/${rid}/views/data`]: { body: data }, [`GET /api/runs/${rid}/views/accuracy`]: { body: viewFixture("accuracy") } },
      (c) => c.querySelector('[aria-label="Data QA"]'),
    );
    const header = container.querySelector('[data-view="data"] > header')!;
    expect(header.querySelector("[title]")).toBeNull();
    const chips = [...header.querySelectorAll("[data-tip-chip]")];
    expect(chips.map((c) => c.textContent)).toEqual(["DEMO", AVAILABILITY_WORDS.stale.label]);
    expect(tipOf(chips[0]).textContent).toBe(term("demo").tip);
    expect(tipOf(chips[1]).textContent).toBe(AVAILABILITY_WORDS.stale.tip);
    m.restore();
  });
});

describe("Model details › Data & checks", () => {
  it("explains every jargon tile, block and chart with an ⓘ beside it", async () => {
    const rid = nextRid("data");
    const { container, m } = await mount(
      `/r/${rid}/details/data`,
      <DataQa />,
      { [`GET /api/runs/${rid}`]: { body: runDetail(rid) }, [`GET /api/runs/${rid}/views/data`]: { body: viewFixture("data") }, [`GET /api/runs/${rid}/views/accuracy`]: { body: viewFixture("accuracy") } },
      (c) => c.querySelector('[aria-label="Data QA"]') && c.querySelector('[aria-label="Forcing"]'),
    );
    // tiles: the labels stay the technical names, each with its ⓘ
    const tile = (label: string) => [...container.querySelectorAll('[aria-label="Data QA"] .kpi')].find((t) => t.querySelector(".label")!.textContent === label)!;
    expect(tile("Grid fill").querySelector("button.infotip")!.getAttribute("data-term")).toBe("grid_fill");
    expect(tile("Rounding noise floor").querySelector("button.infotip")!.getAttribute("data-term")).toBe("noise_floor");
    for (const label of ["Input points", "Dropped (non-finite)", "Clipped values", "Cell collisions", "Background"]) expect(tile(label).querySelector("button.infotip"), label).not.toBeNull();
    // block titles: the ⓘ sits beside the h3, so the heading's name is its title alone
    const block = (title: string) => [...container.querySelectorAll<HTMLElement>("section.card")].find((s) => s.querySelector(":scope > header h3")?.textContent === title)!;
    for (const [title, id] of [
      ["Forcing", "forcing"],
      ["Dose scale", "dose_scale"],
      ["Coarse aggregation", "coarse"],
    ] as const) {
      const head = block(title).querySelector(":scope > header")!;
      expect(head.querySelector("h3 button"), title).toBeNull();
      expect(head.querySelector("button.infotip")!.getAttribute("data-term"), title).toBe(id);
    }
    for (const title of ["QA flags", "Join provenance"]) expect(block(title).querySelector(":scope > header button.infotip"), title).not.toBeNull();
    // the forcing card's radiation terms
    const forcing = container.querySelector('[aria-label="Forcing"]')!;
    expect([...forcing.querySelectorAll("dt button.infotip")].map((b) => b.getAttribute("aria-label"))).toEqual(["What is Shortwave down?", "What is Net longwave?"]);
    // the fractional-part chart says what it is for, with the glossary's ⓘ, and its caption in plain words
    const frac = container.querySelector('figure[data-chart="Fractional part of the target"]')!;
    expect(frac.querySelector(".chart-note")!.textContent).toBe("Were the readings rounded?");
    expect(tipOf(frac.querySelector(".chart-note button.infotip")!).querySelector(".tip-text")!.textContent).toBe(term("fractional_part").tip);
    expect(frac.querySelector("figcaption")!.textContent).toContain("of readings are whole numbers");
    expect(frac.querySelector("figcaption")!.textContent).toContain(`${term("noise_floor").label.toLowerCase()} (rounding-noise floor)`);
    // predictor distributions: the glossary's word for a predictor
    expect(container.querySelector('figure[data-chart="Predictor distributions"] .chart-note button.term')!.getAttribute("data-term")).toBe("predictor");
    expect(container.querySelector('figure[data-chart="Predictor correlations"] .chart-note button.infotip')).not.toBeNull();
    // Model details may keep native titles, but this page has none left
    expect(container.querySelector("[title]")).toBeNull();
    m.restore();
  });
});

describe("Model details sections", () => {
  it("gives a technical section title an ⓘ beside it and writes its intro in plain words with glossary terms", async () => {
    const m = mockFetch({ "GET /api/runs/r1": { body: runDetail("r1") }, "GET /api/runs/r1/views/response": { body: viewFixture("response") } });
    try {
      navigate("/r/r1/details/response", { replace: true });
      const { container } = render(<Details />);
      const h2 = await waitFor(() => container.querySelector<HTMLElement>('.detail-body [data-view="response"] h2'), 10000, "the response page");
      await flush(4);
      expect(h2.textContent).toBe("Response curves");
      expect(h2.querySelector("button")).toBeNull();
      const tip = h2.parentElement!.querySelector<HTMLButtonElement>(":scope > button.infotip")!;
      expect(tip.getAttribute("aria-label")).toBe("What is Response curves?");
      expect(tip.getAttribute("data-term")).toBe("partial_dependence");
      expect(tipOf(tip).querySelector(".tip-tech code")!.textContent).toBe(term("partial_dependence").technical);
      // one ⓘ: the section's, on its first page's title
      expect(container.querySelectorAll(".detail-body h2 + button.infotip")).toHaveLength(1);
      const intro = container.querySelector('.detail-intro[data-intro="response"]')!;
      expect([...intro.querySelectorAll("button.term")].map((b) => b.getAttribute("data-term"))).toEqual(["dose_response"]);
    } finally {
      m.restore();
    }
  });

  it("leaves plain section titles without an ⓘ and names each intro's jargon (Data & checks)", async () => {
    const m = mockFetch({
      "GET /api/runs/r1": { body: runDetail("r1") },
      "GET /api/runs/r1/views/data": { body: viewFixture("data") },
      "GET /api/runs/r1/views/accuracy": { body: viewFixture("accuracy") },
    });
    try {
      navigate("/r/r1/details/data", { replace: true });
      const { container } = render(<Details />);
      const h2 = await waitFor(() => container.querySelector<HTMLElement>('.detail-body [data-view="data"] h2'), 10000, "the data page");
      await flush(4);
      expect(h2.textContent).toBe("Data & checks");
      expect(h2.parentElement!.querySelector(":scope > button.infotip")).toBeNull();
      expect(DETAIL_SECTION_TIPS.data).toBeUndefined();
      const intro = container.querySelector('.detail-intro[data-intro="data"]')!;
      expect([...intro.querySelectorAll("button.term")].map((b) => b.getAttribute("data-term"))).toEqual(["dose_scale", "predictor", "forcing"]);
    } finally {
      m.restore();
    }
  });
});

describe("Provenance and the run record", () => {
  it("Provenance explains each block, and the 'dirty' chip in the glossary's words", async () => {
    const rid = nextRid("prov");
    const { container, m } = await mount(
      `/r/${rid}/details/provenance`,
      <Provenance />,
      {
        [`GET /api/runs/${rid}`]: { body: runDetail(rid) },
        [`GET /api/runs/${rid}/views/provenance`]: { body: viewFixture("provenance") },
        [`GET /api/runs/${rid}/config`]: {
          body: { effective: {}, raw: {}, yaml: "name: synthetic_demo\n", source: "launch", config_dir: "/w/p", vs_project_diff: [], vs_defaults: [] },
        },
        "GET /api/projects/p_1/runs": { body: { items: [runSummary(rid)], next_cursor: null } },
        [`GET /api/runs/${rid}/environment`]: { body: { packages: ["numpy==1.26.4"] } },
      },
      (c) => c.querySelector('[aria-label="Hashes"]') && c.querySelector('[aria-label="Run configuration (YAML)"]'),
    );
    const titled = [...container.querySelectorAll("section.card > header")].filter((h) => h.querySelector("button.infotip")).map((h) => h.querySelector("h3")!.textContent);
    expect(titled).toEqual(expect.arrayContaining(["Hashes", "Environment", "Effective configuration", "launch.json"]));
    const dirty = [...container.querySelectorAll("[data-tip-chip]")].find((c) => c.textContent === "dirty")!;
    expect(dirty.querySelector("[title]")).toBeNull();
    expect(tipOf(dirty).textContent).toBe(term("uncommitted_code").tip);
    m.restore();
  });

  it("the run record (Run log) explains cells and the checkpoint, with no native titles in its header", async () => {
    const rid = nextRid("rec");
    const { container, m } = await mount(
      `/r/${rid}/details/runlog`,
      <RunRecord />,
      { [`GET /api/runs/${rid}`]: { body: runDetail(rid) }, [`GET /api/runs/${rid}/views/overview`]: { body: viewFixture("overview") } },
      (c) => c.querySelector('[aria-label="Run header"]'),
    );
    const header = container.querySelector('[aria-label="Run header"]')!;
    expect([...header.querySelectorAll("dt button.infotip")].map((b) => b.getAttribute("data-term"))).toEqual(["cell", "checkpoint"]);
    const card = header.closest("section.card")!;
    expect(card.querySelector("[title]")).toBeNull();
    const demo = [...card.querySelectorAll("[data-tip-chip]")].find((c) => c.textContent === "DEMO")!;
    expect(tipOf(demo).textContent).toBe(term("demo").tip);
    m.restore();
  });
});
