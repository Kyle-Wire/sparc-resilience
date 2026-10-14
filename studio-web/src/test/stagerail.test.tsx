import { describe, expect, it } from "vitest";
import { lastDoneIndex, StageNum, StageRail, stageNumber, type StageRailStage } from "../components/ui/StageRail";
import { render } from "./render";

const STAGES: StageRailStage[] = [
  { id: "data", label: "Data", status: "done", meta: "points loaded" },
  { id: "fit", label: "Fit", status: "done" },
  { id: "map", label: "Map", status: "running", href: "/map" },
  { id: "report", label: "Report" },
  { id: "a", label: "A", status: "skipped" },
  { id: "b", label: "B", status: "failed" },
  { id: "c", label: "C" },
  { id: "d", label: "D" },
];

describe("StageRail", () => {
  it("renders an ordered list of numbered squares tinted along the ramp", () => {
    const { container } = render(<StageRail stages={STAGES} label="Pipeline" />);
    const ol = container.querySelector("ol.stage-rail")!;
    expect(ol.getAttribute("aria-label")).toBe("Pipeline");
    expect(ol.getAttribute("data-orientation")).toBe("vertical");
    const items = [...ol.querySelectorAll(":scope > li")];
    expect(items).toHaveLength(8);
    expect(items.map((li) => li.querySelector(".stage-num")!.textContent)).toEqual(["01", "02", "03", "04", "05", "06", "07", "08"]);
    // the ramp is sampled once across the stages: first purple, last gold, never wrapping
    expect(items.map((li) => li.getAttribute("data-tone"))).toEqual(["0", "1", "2", "3", "3", "4", "5", "6"]);
    expect((items[0] as HTMLElement).style.getPropertyValue("--stage-c")).toBe("var(--tone-0)");
    expect((items[7] as HTMLElement).style.getPropertyValue("--stage-c")).toBe("var(--tone-6)");
    expect((items[7].querySelector(".stage-num") as HTMLElement).style.getPropertyValue("--stage-c")).toBe("var(--tone-6)");
    for (const li of items) expect(li.querySelector(".stage-num")!.getAttribute("aria-hidden")).toBe("true");
    expect(items.map((li) => li.getAttribute("data-status"))).toEqual(["done", "done", "running", "todo", "skipped", "failed", "todo", "todo"]);
  });

  it("fills the rail up to the last done stage and marks the current one", () => {
    const { container } = render(<StageRail stages={STAGES} current="map" orientation="horizontal" />);
    const items = [...container.querySelectorAll("ol.stage-rail > li")];
    // segment after stage 01 is filled (01 → 02); 02 is the last done stage
    expect(items.map((li) => li.hasAttribute("data-filled"))).toEqual([true, false, false, false, false, false, false, false]);
    expect(items.filter((li) => li.getAttribute("aria-current") === "step").map((li) => li.textContent)).toEqual([expect.stringContaining("Map")]);
    expect(container.querySelector("ol")!.getAttribute("data-orientation")).toBe("horizontal");
  });

  it("states status in words, links stages with an href and takes custom content and data attributes", () => {
    const { container } = render(
      <StageRail
        stages={[
          ...STAGES.slice(0, 3),
          { id: "x", label: "X", data: { stage: "S4", locked: undefined }, content: <button type="button">pick</button> },
        ]}
      />,
    );
    const items = [...container.querySelectorAll("ol.stage-rail > li")];
    expect(items[0].textContent).toContain("(done)");
    expect(items[0].querySelector(".stage-rail-meta")!.textContent).toBe("points loaded");
    expect(items[2].querySelector("a")!.getAttribute("href")).toBe("/map");
    expect(items[3].getAttribute("data-stage")).toBe("S4");
    expect(items[3].hasAttribute("data-locked")).toBe(false);
    expect(items[3].querySelector(".stage-rail-content button")!.textContent).toBe("pick");
  });

  it("numbers and finds the last done stage", () => {
    expect(stageNumber(0)).toBe("01");
    expect(stageNumber(10)).toBe("11");
    expect(lastDoneIndex(STAGES)).toBe(1);
    expect(lastDoneIndex([{ status: "todo" }])).toBe(-1);
    const { container } = render(<StageNum index={8} count={11} status="done" filled />);
    const sq = container.querySelector(".stage-num")!;
    expect(sq.textContent).toBe("09");
    expect(sq.getAttribute("data-tone")).toBe("5");
    expect(sq.hasAttribute("data-filled")).toBe(true);
  });
});
