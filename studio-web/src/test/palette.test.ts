import { describe, expect, it } from "vitest";
import golden from "./fixtures/palette.json";
import { RAMPS, getLuts, lut, lutHex, lutToUint32, packRgba, rampColor, resolveCssVars, unpackRgba } from "../theme/palette";

type Ramp = keyof typeof RAMPS;
const names = Object.keys(golden.golden) as Ramp[];

describe("OKLab LUT (SPEC §6.3 golden table)", () => {
  it("ramps match the spec stops", () => {
    for (const n of names) expect([...RAMPS[n]]).toEqual(golden.ramps[n]);
  });
  for (const n of names) {
    it(`${n} entries 0/64/128/191/255 equal palette.json`, () => {
      const table = lut(RAMPS[n], golden.n);
      expect(table.length).toBe(golden.n * 3);
      expect(golden.indices.map((i) => lutHex(table, i))).toEqual(golden.golden[n]);
    });
  }
  it("endpoints reproduce the stops exactly", () => {
    for (const n of names) {
      const t = lut(RAMPS[n]);
      expect(lutHex(t, 0)).toBe(RAMPS[n][0]);
      expect(lutHex(t, 255)).toBe(RAMPS[n][RAMPS[n].length - 1]);
    }
  });
  it("theme LUTs and packed words agree", () => {
    const l = getLuts(false);
    expect(lutHex(l.seq, 64)).toBe("#85b6f0");
    expect(lutHex(getLuts(true).div, 128)).toBe("#393835");
    const w = lutToUint32(l.div);
    expect(unpackRgba(w[128])).toEqual([0xf0, 0xee, 0xeb, 255]);
    expect(packRgba(1, 2, 3, 4)).toBe(0x04030201);
    expect(rampColor("seq", 0.5, false)).toBe(lutHex(l.seq, 128));
  });
  it("resolves CSS variables for exports", () => {
    expect(resolveCssVars("fill:var(--accent);stroke:var(--s2)", false)).toBe("fill:#2a78d6;stroke:#eb6834");
    expect(resolveCssVars("fill:var(--accent)", true)).toBe("fill:#3987e5");
  });
});
