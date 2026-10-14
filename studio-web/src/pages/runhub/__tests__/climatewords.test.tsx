// Future climates and heat-risk levels in plain words (pages/runhub/climateWords.tsx): emissions
// futures and periods by their glossary names, the pathway code kept for a tooltip's technical
// line, server strings that name a future by code rewritten, and the heat-risk levels' ⓘ.
import { act } from "react";
import { describe, expect, it } from "vitest";
import { heatCategory } from "../../../content/glossary";
import { render } from "../../../test/render";
import { emissionsName, futureName, futureShort, futureTechnical, FutureName, HeatLevelsTip, pathwayCodes, periodName, plainCase } from "../climateWords";

const tooltip = () => document.querySelector<HTMLElement>('[role="tooltip"]:not([hidden])');

describe("future climates in plain words", () => {
  it("names a future by its emissions and period, the code kept for the technical line", () => {
    expect(futureName("ssp245", "2041-2060")).toBe("Moderate emissions, mid-century (2041–2060)");
    expect(futureName("SSP5-8.5", "2081-2100")).toBe("Very high emissions, late century (2081–2100)");
    expect(futureName("ssp126", "2061-2080")).toBe("Low emissions, 2061–2080");
    expect(futureName("mystery", "2041-2060", "Mystery pathway")).toBe("Mystery pathway, mid-century (2041–2060)");
    expect(futureTechnical("ssp245", "2041-2060")).toBe("SSP2-4.5, 2041–2060");
    expect(emissionsName("ssp370")).toBe("High emissions");
    expect(periodName("2021-2040")).toBe("Near-term (2021–2040)");
    expect(pathwayCodes(["ssp245", "ssp585"])).toBe("SSP2-4.5 = moderate emissions · SSP5-8.5 = very high emissions");
  });

  it("has a compact form for chart rows (labels there are cut at about 26 characters)", () => {
    expect(futureShort("ssp245", "2041-2060")).toBe("Moderate, mid-century");
    expect(futureShort("ssp585", "2081-2100")).toBe("Very high, late century");
    expect(futureShort("ssp460", "2041-2060")).toBe("Moderate-high, mid-century");
    expect(futureShort("ssp245", "2061-2080")).toBe("Moderate, 2061–2080");
    // the main CMIP6 pathways fit whole (SSP4-6.0 late century, "Moderate-high, late century", is one over)
    for (const e of ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"])
      for (const p of ["2021-2040", "2041-2060", "2081-2100"]) expect(futureShort(e, p).length, `${e} ${p}`).toBeLessThanOrEqual(26);
  });

  it("rewrites server strings that name a future by code, keeping the rest", () => {
    expect(plainCase("SSP2-4.5 2041–2060")).toBe("Moderate emissions, mid-century (2041–2060)");
    expect(plainCase("SSP2-4.5 2041-2060 (with the cooling package)")).toBe("Moderate emissions, mid-century (2041–2060) (with the cooling package)");
    expect(plainCase("ssp585:2081-2100", true)).toBe("Very high, late century");
    expect(plainCase("Today")).toBe("Today");
    expect(plainCase("Zone 2041")).toBe("Zone 2041");
  });

  it("FutureName explains the emissions words in a tooltip with the pathway code", () => {
    const { container } = render(<FutureName experiment="ssp245" period="2041-2060" />);
    expect(container.textContent).toBe("Moderate emissions, mid-century (2041–2060)");
    const btn = container.querySelector<HTMLButtonElement>('button.term[data-term="ssp"]')!;
    act(() => btn.focus());
    expect(tooltip()!.textContent).toMatch(/2\.7 °C/);
    expect(tooltip()!.querySelector(".tip-tech")!.textContent).toBe("Technical: SSP2-4.5, 2041–2060");
    act(() => btn.blur());
  });
});

describe("heat-risk levels", () => {
  it("the ⓘ lists each level with its feels-like range", () => {
    const { container } = render(<HeatLevelsTip />);
    const btn = container.querySelector<HTMLButtonElement>("button.infotip")!;
    expect(btn.getAttribute("aria-label")).toBe("What is Heat-risk level?");
    act(() => btn.focus());
    const text = tooltip()!.textContent ?? "";
    for (const k of ["caution", "extreme_caution", "danger", "extreme_danger"] as const) expect(text).toContain(`${heatCategory(k).label}: ${heatCategory(k).tip}`);
    expect(text).toContain("90–103 °F");
    act(() => btn.blur());
  });
});
