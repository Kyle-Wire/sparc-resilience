// PlanGraph (SPEC §5.4): node states, reasons and estimates of a fixture plan, with S4 marked
// "required by S6"; state is carried by text and icon, not colour alone.
import { describe, expect, it } from "vitest";
import { render } from "../../../test/render";
import { PlanGraph } from "../components/PlanGraph";
import { planNodes } from "../__fixtures__/api";

describe("PlanGraph", () => {
  it("renders every node in stage order with its state and reason", () => {
    const { container } = render(<PlanGraph nodes={planNodes()} total={{ lo: 540, hi: 780 }} />);
    const nodes = [...container.querySelectorAll<HTMLElement>("li.plan-node")];
    expect(nodes.map((n) => n.dataset.node)).toEqual(["S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7", "finish"]);
    const by = (id: string) => container.querySelector<HTMLElement>(`li.plan-node[data-node="${id}"]`)!;
    expect(by("S0").dataset.state).toBe("will_run");
    expect(by("S1").dataset.state).toBe("cached");
    expect(by("baselines").dataset.state).toBe("skipped");

    const s4 = by("S4");
    expect(s4.dataset.state).toBe("will_run");
    expect(s4.querySelector(".plan-state")!.textContent).toBe("will run");
    expect(s4.querySelector(".plan-reason")!.textContent).toContain("required by S6");
    expect(s4.querySelector(".plan-units")!.textContent).toContain("22 engine passes");
    expect(s4.querySelector(".plan-est")!.textContent).toBe("≈4–6 min");

    expect(by("S1").querySelector(".plan-state")!.textContent).toBe("cached");
    expect(by("S1").textContent).toContain("from the checkpoint");
    expect(by("baselines").textContent).toContain("disabled in the config (cv.baselines)");
    expect(by("S5").textContent).toContain("not requested");
    expect(by("climate").textContent).toContain("needs S5 (scenarios)");
    expect(by("S7").textContent).toContain("no budget set");
    // skipped and cached nodes carry no estimate
    expect(by("S5").querySelector(".plan-est")!.textContent).toBe("");
    expect(by("S1").querySelector(".plan-est")!.textContent).toBe("");
  });

  it("summarises the counts and the total range", () => {
    const { container } = render(<PlanGraph nodes={planNodes()} total={{ lo: 540, hi: 780 }} />);
    const cap = container.querySelector("figcaption")!.textContent!;
    expect(cap).toContain("5 will run");
    expect(cap).toContain("5 skipped");
    expect(cap).toContain("1 cached");
    expect(cap).toContain("total ≈9–13 min");
  });
});
