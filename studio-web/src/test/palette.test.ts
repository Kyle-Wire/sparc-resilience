import { describe, expect, it } from "vitest";
import golden from "./fixtures/palette.json";
import { RAMPS, THEME_COLORS, catColors, getLuts, hexToLinear, linearToLab, lut, lutHex, lutToUint32, packRgba, rampColor, resolveCssVars, stageTone, stageToneIndex, toLab, unpackRgba } from "../theme/palette";

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
    expect(lutHex(l.seq, 64)).toBe("#eca32b");
    expect(lutHex(getLuts(true).div, 128)).toBe("#3b3134");
    const w = lutToUint32(l.div);
    expect(unpackRgba(w[128])).toEqual([0xe9, 0xe2, 0xd5, 255]);
    expect(packRgba(1, 2, 3, 4)).toBe(0x04030201);
    expect(rampColor("seq", 0.5, false)).toBe(lutHex(l.seq, 128));
  });
  it("resolves CSS variables for exports", () => {
    expect(resolveCssVars("fill:var(--accent);stroke:var(--s1)", false)).toBe("fill:#e73c25;stroke:#602468");
    expect(resolveCssVars("fill:var(--accent)", true)).toBe("fill:#f0553c");
    // stage tones between stops are OKLab color-mixes, flattened to hex for exports
    expect(resolveCssVars(`fill:${stageTone(1, 11)}`, false)).toMatch(/^fill:#[0-9a-f]{6}$/);
  });
});

describe("categorical and stage colours", () => {
  it("six distinct categorical colours per theme, clear of the diverging midpoint", () => {
    for (const dark of [false, true]) {
      const { colors } = catColors(dark);
      expect(colors).toHaveLength(6);
      expect(new Set(colors).size).toBe(6);
      const labs = colors.map(toLab);
      for (let i = 0; i < labs.length; i++)
        for (let j = i + 1; j < labs.length; j++) expect(Math.hypot(...labs[i].map((v, k) => v - labs[j][k])), `${colors[i]} vs ${colors[j]}`).toBeGreaterThan(0.1);
    }
  });
  it("the diverging midpoint differs from the surface and page colours", () => {
    for (const [ramp, t] of [["divLight", THEME_COLORS.light], ["divDark", THEME_COLORS.dark]] as const) {
      const mid = toLab(RAMPS[ramp][3]);
      for (const bg of [t.surface, t.page]) expect(Math.abs(mid[0] - toLab(bg)[0]), `${ramp} vs ${bg}`).toBeGreaterThan(0.05);
    }
  });
  it("stageTone spans the purple→gold ramp once across n stages", () => {
    expect(stageTone(0, 11)).toBe("var(--tone-0)");
    expect(stageTone(10, 11)).toBe("var(--tone-6)");
    expect(stageTone(5, 11)).toBe("var(--tone-3)");
    expect(stageTone(1, 11)).toBe("color-mix(in oklab, var(--tone-0) 40%, var(--tone-1))");
    expect(stageTone(0, 1)).toBe("var(--tone-0)");
    const all = Array.from({ length: 11 }, (_, i) => stageTone(i, 11));
    expect(new Set(all).size).toBe(11);
    expect(Array.from({ length: 11 }, (_, i) => stageToneIndex(i, 11))).toEqual([0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6]);
  });
});

describe("ramp perception", () => {
  const L = (hex: string) => toLab(hex)[0];
  it("sequential ramps have monotonic OKLab lightness (stops and every LUT entry)", () => {
    for (const [name, dir] of [["seqLight", -1], ["seqDark", 1]] as const) {
      const stops = RAMPS[name].map(L);
      for (let i = 1; i < stops.length; i++) expect(Math.sign(stops[i] - stops[i - 1]), `${name} stop ${i}`).toBe(dir);
      const t = lut(RAMPS[name]);
      for (let i = 1; i < 256; i++) expect((L(lutHex(t, i)) - L(lutHex(t, i - 1))) * dir, `${name} entry ${i}`).toBeGreaterThanOrEqual(-1e-3);
    }
  });

  // Machado, Oliveira & Fernandes (2009), severity 1.0, applied to linear-light sRGB.
  const MACHADO: Record<string, number[][]> = {
    protan: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    deutan: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]],
    tritan: [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.3039]],
  };
  const simLab = (hex: string, m: number[][] | null) => {
    const v = hexToLinear(hex);
    const out = m ? (m.map((r) => Math.min(1, Math.max(0, r[0] * v[0] + r[1] * v[1] + r[2] * v[2]))) as [number, number, number]) : v;
    return linearToLab(out);
  };
  const dE = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  for (const name of ["divLight", "divDark"] as const) {
    it(`${name} ends stay distinguishable under protan/deutan/tritan simulation (OKLab ΔE ≥ 0.15)`, () => {
      const r = RAMPS[name];
      for (const [kind, m] of [["normal", null], ...Object.entries(MACHADO)] as [string, number[][] | null][])
        expect(dE(simLab(r[0], m), simLab(r[r.length - 1], m)), `${name} ${kind}`).toBeGreaterThanOrEqual(0.15);
    });
  }
  it("diverging ramps put the cool (purple, b < 0) arm at the low end", () => {
    for (const name of ["divLight", "divDark"] as const) {
      const r = RAMPS[name];
      expect(toLab(r[0])[2], `${name} low end is blue (b < 0)`).toBeLessThan(0);
      expect(toLab(r[r.length - 1])[2], `${name} high end is warm (b > 0)`).toBeGreaterThan(0);
    }
  });
});
