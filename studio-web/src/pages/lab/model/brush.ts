// Brush edit layers (SPEC §7.5, §12.5): one Float32 edit array per lever, aligned with run
// rows. Painting goes through the map kit's applyBrush, so a stroke OVERWRITES the cells it
// touches with clamp(base + amount) − base computed against the exact Float32 inputs:
// repainting the same cells never accumulates and base + edit never leaves the lever bounds.
// Layers snapshot to a sparse form for undo, travel to the preview as SparseEdit payloads
// and upload as edit blobs (Int32 idx then Float32 val) when the draft is saved.
import { denseToSparse, editBlobBody, encodeSparseEdit } from "../../../api/binary";
import type { SparseEdit } from "../../../api/types";
import { applyBrush, type BrushSettings } from "../../../map/tools/brush";

export type SparseLayer = { idx: Int32Array; val: Float32Array };

/** Sparse copy of every non-empty lever layer (cheap to keep in undo history). */
export type BrushSnapshot = Record<string, SparseLayer>;

export type BrushStroke = { mode: "add" | "erase"; amount: number; base: Float32Array; bounds: [number, number] };

export class BrushLayers {
  readonly n: number;
  private layers = new Map<string, Float32Array>();
  private versions = new Map<string, number>();

  constructor(n: number) {
    this.n = n;
  }

  /** The live edit array of a lever (created empty on first use; the map tool paints into it). */
  array(lever: string): Float32Array {
    let a = this.layers.get(lever);
    if (!a) {
      a = new Float32Array(this.n);
      this.layers.set(lever, a);
    }
    return a;
  }

  /** Bumped whenever a lever's layer changes (upload caching). */
  version(lever: string): number {
    return this.versions.get(lever) ?? 0;
  }

  touch(lever: string): void {
    this.versions.set(lever, this.version(lever) + 1);
  }

  /** Paint (or erase) rows of one lever; returns the rows whose edit changed. */
  paint(lever: string, rows: number[], stroke: BrushStroke, radius_m = 0): number[] {
    const settings: BrushSettings = { amount: stroke.amount, mode: stroke.mode, radius_m, base: stroke.base, bounds: stroke.bounds };
    const changed = applyBrush(this.array(lever), rows, settings);
    if (changed.length) this.touch(lever);
    return changed;
  }

  /** Levers with at least one edited cell. */
  levers(): string[] {
    return [...this.layers.entries()].filter(([, a]) => a.some((v) => v !== 0)).map(([k]) => k);
  }

  editedCount(lever: string): number {
    const a = this.layers.get(lever);
    if (!a) return 0;
    let c = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== 0) c++;
    return c;
  }

  isEmpty(): boolean {
    return this.levers().length === 0;
  }

  clear(lever?: string): void {
    for (const [k, a] of this.layers) {
      if (lever !== undefined && k !== lever) continue;
      if (a.some((v) => v !== 0)) {
        a.fill(0);
        this.touch(k);
      }
    }
  }

  snapshot(): BrushSnapshot {
    const out: BrushSnapshot = {};
    for (const [k, a] of this.layers) {
      const s = denseToSparse(a);
      if (s.idx.length) out[k] = s;
    }
    return out;
  }

  /**
   * Restore a snapshot IN PLACE (the arrays keep their identity, so a map tool holding one
   * keeps painting into the restored layer).
   */
  restore(snap: BrushSnapshot): void {
    const levers = new Set([...this.layers.keys(), ...Object.keys(snap)]);
    for (const k of levers) {
      const a = this.array(k);
      const s = snap[k];
      const before = a.some((v) => v !== 0) || !!s;
      a.fill(0);
      if (s) for (let i = 0; i < s.idx.length; i++) a[s.idx[i]] = s.val[i];
      if (before) this.touch(k);
    }
  }

  /** The `brush` field of a preview request: `{lever: {idx, val}}` for every non-empty lever. */
  previewPayload(): Record<string, SparseEdit> {
    const out: Record<string, SparseEdit> = {};
    for (const [k, a] of this.layers) {
      const s = denseToSparse(a);
      if (s.idx.length) out[k] = encodeSparseEdit(s.idx, s.val);
    }
    return out;
  }

  /** Raw edit-blob body of one lever (`PUT /blobs?kind=edit`). */
  blobBody(lever: string): { body: Uint8Array; count: number } {
    const s = denseToSparse(this.array(lever));
    return editBlobBody(s.idx, s.val);
  }
}

/** Snapshots equal cell for cell (undo coalescing and tests). */
export function snapshotsEqual(a: BrushSnapshot, b: BrushSnapshot): boolean {
  const ka = Object.keys(a).sort();
  const kb = Object.keys(b).sort();
  if (ka.join("\u0000") !== kb.join("\u0000")) return false;
  for (const k of ka) {
    const x = a[k];
    const y = b[k];
    if (x.idx.length !== y.idx.length) return false;
    for (let i = 0; i < x.idx.length; i++) if (x.idx[i] !== y.idx[i] || x.val[i] !== y.val[i]) return false;
  }
  return true;
}

/**
 * Lever bounds for brushing: the configured min/max, else unbounded on that side (the engine
 * then applies its own support rule).
 */
export function leverBounds(lever: { min: number | null; max: number | null } | undefined): [number, number] {
  return [lever?.min ?? -Infinity, lever?.max ?? Infinity];
}
