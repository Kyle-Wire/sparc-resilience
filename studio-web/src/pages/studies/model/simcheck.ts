// Simulation-check grid model (SPEC §5.11, §8): generator × seed cells from the study view,
// coloured by effect share on the diverging ramp centred on 1 (recovered exactly), grey while
// pending or running, red on error, with a dot for a replicate whose first draw failed the
// data-matching gate and was redrawn.
import type { SimCellStatus, SimcheckCell, SimcheckView } from "../../../api/studies";
import { rampColor, TOKEN_VALUES } from "../../../theme/palette";

export type SimGridCell = SimcheckCell & { planned: boolean };

export type SimGrid = {
  generators: string[];
  seeds: number[];
  cells: Map<string, SimGridCell>;
  /** Half-width of the share scale around 1 (at least 1, so 0 and 2 are the ramp ends). */
  span: number;
};

const STATUSES: readonly SimCellStatus[] = ["pending", "running", "done", "gate_fail", "error"];

export const cellKey = (generator: string, seed: number) => `${generator}/${seed}`;

/**
 * The grid of a simcheck view: the design's generators (count > 0, in design order) plus any
 * generator the rows add; seeds 0…max−1; a design cell with no row yet is pending.
 */
export function simGrid(view: SimcheckView | null | undefined): SimGrid {
  const design = view?.design ?? {};
  const generators = Object.keys(design).filter((g) => Number(design[g]) > 0);
  let nSeeds = Math.max(0, ...generators.map((g) => Math.round(Number(design[g]) || 0)));
  const cells = new Map<string, SimGridCell>();
  for (const c of view?.grid ?? []) {
    if (!generators.includes(c.generator)) generators.push(c.generator);
    nSeeds = Math.max(nSeeds, c.seed + 1);
    const status = STATUSES.includes(c.status) ? c.status : "pending";
    cells.set(cellKey(c.generator, c.seed), { ...c, status, planned: true });
  }
  for (const g of generators) {
    const n = Math.round(Number(design[g]) || 0);
    for (let s = 0; s < n; s++) {
      const k = cellKey(g, s);
      if (!cells.has(k)) cells.set(k, { generator: g, seed: s, status: "pending", share: null, ci_covers: null, causal_covers: null, seconds: null, gate_attempt: null, planned: true });
    }
  }
  const shares = [...cells.values()].map((c) => c.share).filter((v): v is number => v !== null && Number.isFinite(v));
  const span = Math.max(1, ...shares.map((v) => Math.abs(v - 1)));
  return { generators, seeds: Array.from({ length: nSeeds }, (_, i) => i), cells, span };
}

/** Position of a share on the diverging ramp: 0.5 at share 1, 0 at 1 − span, 1 at 1 + span. */
export function shareT(share: number, span: number): number {
  return Math.min(1, Math.max(0, 0.5 + (0.5 * (share - 1)) / (span || 1)));
}

export type SimCellLook = {
  /** Background colour (always set: ramp colour, grey or red). */
  fill: string;
  /** Text marker: "•" for a gate redraw, "!" for an error, "…" while running, "–" for a finished replicate without a share. */
  marker: string;
  redraw: boolean;
  label: string;
  /** Marker colour readable on `fill`. */
  ink: string;
};

/** Dark or light text for a background colour (#rgb or #rrggbb; anything else gets dark text). */
export function inkOn(fill: string): string {
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(fill.trim());
  if (!m) return "#111111";
  const h = m[1].length === 3 ? m[1].replace(/./g, (c) => c + c) : m[1];
  const lin = (i: number) => {
    const c = parseInt(h.slice(i, i + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  const lum = 0.2126 * lin(0) + 0.7152 * lin(2) + 0.0722 * lin(4);
  return lum > 0.22 ? "#111111" : "#ffffff";
}

export function simCellLook(cell: SimGridCell | undefined, generator: string, seed: number, span: number, dark: boolean): SimCellLook {
  const tok = TOKEN_VALUES[dark ? "dark" : "light"];
  const grey = tok["--gray-mark"];
  const red = tok["--critical"];
  const status: SimCellStatus = cell?.status ?? "pending";
  const share = cell?.share ?? null;
  const redraw = (cell?.gate_attempt ?? 0) > 0;
  const finished = status === "done" || status === "gate_fail";
  const hasShare = share !== null && Number.isFinite(share) && finished;
  // A finished replicate without a share (the null generator plants no effect) is neutral, not pending.
  const fill = status === "error" ? red : hasShare ? rampColor("div", shareT(share as number, span), dark) : finished ? tok["--nodata"] : grey;
  const marker = status === "error" ? "!" : redraw ? "•" : status === "running" ? "…" : finished && !hasShare ? "–" : "";
  const covers = cell?.ci_covers === true ? ", interval covers the truth" : cell?.ci_covers === false ? ", interval misses the truth" : "";
  const what =
    status === "error"
      ? "error"
      : status === "running"
        ? "running"
        : status === "pending"
          ? "pending"
          : `${status === "gate_fail" ? "failed the data gate, " : ""}${hasShare ? `share ${(share as number).toFixed(2)}` : "no planted effect to share"}${covers}`;
  const label = `${generator}, seed ${seed}: ${what}${redraw ? `, gate redraw (attempt ${(cell?.gate_attempt ?? 0) + 1})` : ""}`;
  return { fill, marker, redraw, label, ink: inkOn(fill) };
}

/** Done shares per generator (for the strip plot), in grid order. */
export function sharesByGenerator(grid: SimGrid): { generator: string; shares: number[] }[] {
  return grid.generators.map((g) => ({
    generator: g,
    shares: grid.seeds
      .map((s) => grid.cells.get(cellKey(g, s)))
      .filter((c): c is SimGridCell => !!c && c.status === "done" && c.share !== null && Number.isFinite(c.share))
      .map((c) => c.share as number),
  }));
}

/** Counts by status, and the ETA the view gives (or mean seconds × remaining ÷ workers). */
export function simProgress(grid: SimGrid, workers = 1, etaFromView: number | null = null): { done: number; total: number; errors: number; running: number; eta_s: number | null } {
  const all = [...grid.cells.values()];
  const done = all.filter((c) => c.status === "done" || c.status === "gate_fail").length;
  const errors = all.filter((c) => c.status === "error").length;
  const running = all.filter((c) => c.status === "running").length;
  const secs = all.map((c) => c.seconds).filter((v): v is number => v !== null && Number.isFinite(v));
  const remaining = Math.max(0, all.length - done - errors);
  const eta = etaFromView ?? (secs.length && remaining ? ((secs.reduce((a, b) => a + b, 0) / secs.length) * remaining) / Math.max(1, workers) : remaining ? null : 0);
  return { done, total: all.length, errors, running, eta_s: eta };
}

export function quantiles(xs: number[]): [number, number, number, number, number] | null {
  if (!xs.length) return null;
  const v = [...xs].sort((a, b) => a - b);
  const q = (p: number) => {
    const pos = p * (v.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, v.length - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
  };
  return [v[0], q(0.25), q(0.5), q(0.75), v[v.length - 1]];
}
