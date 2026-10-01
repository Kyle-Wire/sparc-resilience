// Binary wire formats (api.md §0.4): typed array views, packed arrays with X-SPARC-Offsets,
// selection bitsets (LSB-first, base64) and sparse brush edits.
//
// All arrays on the wire are little-endian. Every browser Studio supports is little-endian,
// but the views below still check and byte-swap on a big-endian host rather than misread.

export type Dtype = "float32" | "uint8" | "int16" | "int32" | "int64";
export type TypedArray = Float32Array | Uint8Array | Int16Array | Int32Array | BigInt64Array;

export type OffsetEntry = { name: string; dtype: Dtype; offset: number; length: number };

const BYTES: Record<Dtype, number> = { float32: 4, uint8: 1, int16: 2, int32: 4, int64: 8 };

export const IS_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export function bytesPerElement(dtype: Dtype): number {
  return BYTES[dtype];
}

function swapCopy(buf: ArrayBuffer, byteOffset: number, length: number, size: number): ArrayBuffer {
  const src = new Uint8Array(buf, byteOffset, length * size);
  const out = new Uint8Array(length * size);
  for (let i = 0; i < length; i++) for (let b = 0; b < size; b++) out[i * size + b] = src[i * size + (size - 1 - b)];
  return out.buffer;
}

/**
 * A typed view over little-endian data. Aligned data on a little-endian host is a zero-copy
 * view; unaligned offsets (or a big-endian host) are copied.
 */
export function viewOf(buf: ArrayBuffer, dtype: Dtype, byteOffset = 0, length?: number): TypedArray {
  const size = BYTES[dtype];
  const n = length ?? Math.floor((buf.byteLength - byteOffset) / size);
  if (byteOffset + n * size > buf.byteLength) throw new RangeError(`array ${dtype}[${n}] at ${byteOffset} overruns ${buf.byteLength} bytes`);
  let b = buf;
  let off = byteOffset;
  if (size > 1 && (!IS_LITTLE_ENDIAN || off % size !== 0)) {
    b = IS_LITTLE_ENDIAN ? buf.slice(off, off + n * size) : swapCopy(buf, off, n, size);
    off = 0;
  }
  switch (dtype) {
    case "float32":
      return new Float32Array(b, off, n);
    case "uint8":
      return new Uint8Array(b, off, n);
    case "int16":
      return new Int16Array(b, off, n);
    case "int32":
      return new Int32Array(b, off, n);
    case "int64":
      return new BigInt64Array(b, off, n);
  }
}

export const asFloat32 = (buf: ArrayBuffer, offset = 0, length?: number) => viewOf(buf, "float32", offset, length) as Float32Array;
export const asUint8 = (buf: ArrayBuffer, offset = 0, length?: number) => viewOf(buf, "uint8", offset, length) as Uint8Array;
export const asInt32 = (buf: ArrayBuffer, offset = 0, length?: number) => viewOf(buf, "int32", offset, length) as Int32Array;
export const asInt64 = (buf: ArrayBuffer, offset = 0, length?: number) => viewOf(buf, "int64", offset, length) as BigInt64Array;

/** Int64 ids as JS numbers (exact up to 2^53). */
export function int64ToNumbers(a: BigInt64Array): number[] {
  const out = new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Number(a[i]);
  return out;
}

const DTYPES = new Set<Dtype>(["float32", "uint8", "int16", "int32", "int64"]);

export function isDtype(v: unknown): v is Dtype {
  return typeof v === "string" && DTYPES.has(v as Dtype);
}

/** Parse the X-SPARC-Offsets header: `[{"name","dtype","offset","length"}, …]`. */
export function parseOffsets(header: string): OffsetEntry[] {
  let raw: unknown;
  try {
    raw = JSON.parse(header);
  } catch {
    throw new Error("X-SPARC-Offsets is not valid JSON");
  }
  if (!Array.isArray(raw)) throw new Error("X-SPARC-Offsets must be a JSON array");
  return raw.map((e, i) => {
    const o = e as Partial<OffsetEntry>;
    if (typeof o.name !== "string" || !DTYPES.has(o.dtype as Dtype) || !Number.isInteger(o.offset) || !Number.isInteger(o.length) || (o.offset as number) < 0 || (o.length as number) < 0) {
      throw new Error(`X-SPARC-Offsets entry ${i} is malformed`);
    }
    return { name: o.name, dtype: o.dtype as Dtype, offset: o.offset as number, length: o.length as number };
  });
}

/** Split a packed body into named typed views. */
export function unpack(buf: ArrayBuffer, offsets: OffsetEntry[]): Record<string, TypedArray> {
  const out: Record<string, TypedArray> = {};
  for (const e of offsets) out[e.name] = viewOf(buf, e.dtype, e.offset, e.length);
  return out;
}

/** Build a packed body (each array 8-byte aligned) and its offsets; used by tests and uploads. */
export function pack(arrays: { name: string; dtype: Dtype; data: TypedArray }[]): { buffer: ArrayBuffer; offsets: OffsetEntry[] } {
  const offsets: OffsetEntry[] = [];
  let pos = 0;
  for (const a of arrays) {
    pos = Math.ceil(pos / 8) * 8;
    offsets.push({ name: a.name, dtype: a.dtype, offset: pos, length: a.data.length });
    pos += a.data.length * BYTES[a.dtype];
  }
  const out = new Uint8Array(Math.ceil(pos / 8) * 8);
  arrays.forEach((a, i) => {
    const src = new Uint8Array(a.data.buffer, a.data.byteOffset, a.data.byteLength);
    if (IS_LITTLE_ENDIAN || BYTES[a.dtype] === 1) out.set(src, offsets[i].offset);
    else out.set(new Uint8Array(swapCopy(a.data.buffer as ArrayBuffer, a.data.byteOffset, a.data.length, BYTES[a.dtype])), offsets[i].offset);
  });
  return { buffer: out.buffer, offsets };
}

// ---------------------------------------------------------------- base64

export function bytesToBase64(bytes: Uint8Array): string {
  let s = "";
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) s += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + CH)));
  return btoa(s);
}

export function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64.replace(/-/g, "+").replace(/_/g, "/").replace(/\s+/g, ""));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// ---------------------------------------------------------------- bitsets

/**
 * Pack a per-row mask into LSB-first bytes: byte i holds rows 8i … 8i+7, row 8i in bit 0.
 * `mask[r]` is truthy for selected rows.
 */
export function packBits(mask: ArrayLike<number | boolean>): Uint8Array {
  const n = mask.length;
  const out = new Uint8Array(Math.ceil(n / 8));
  for (let r = 0; r < n; r++) if (mask[r]) out[r >> 3] |= 1 << (r & 7);
  return out;
}

/** Unpack LSB-first bytes into a 0/1 Uint8Array of length n. */
export function unpackBits(bytes: Uint8Array, n: number): Uint8Array {
  if (bytes.length < Math.ceil(n / 8)) throw new RangeError(`bitset has ${bytes.length} bytes, needs ${Math.ceil(n / 8)} for ${n} rows`);
  const out = new Uint8Array(n);
  for (let r = 0; r < n; r++) out[r] = (bytes[r >> 3] >> (r & 7)) & 1;
  return out;
}

/** Encode a mask as the wire `Bitset` (base64 of the packed bytes). */
export function encodeBitset(mask: ArrayLike<number | boolean>): string {
  return bytesToBase64(packBits(mask));
}

/** Decode a wire `Bitset` into a 0/1 mask of length n. */
export function decodeBitset(b64: string, n: number): Uint8Array {
  return unpackBits(base64ToBytes(b64), n);
}

export function countBits(mask: ArrayLike<number | boolean>): number {
  let c = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) c++;
  return c;
}

// ---------------------------------------------------------------- sparse edits

function leBytes(a: Int32Array | Float32Array): Uint8Array {
  const out = new Uint8Array(a.length * 4);
  const dv = new DataView(out.buffer);
  if (a instanceof Int32Array) for (let i = 0; i < a.length; i++) dv.setInt32(i * 4, a[i], true);
  else for (let i = 0; i < a.length; i++) dv.setFloat32(i * 4, a[i], true);
  return out;
}

/** Rows with a non-zero edit, as the JSON `SparseEdit` {idx, val} (api.md §0.4). */
export function encodeSparseEdit(idx: Int32Array, val: Float32Array): { idx: string; val: string } {
  if (idx.length !== val.length) throw new RangeError("idx and val lengths differ");
  return { idx: bytesToBase64(leBytes(idx)), val: bytesToBase64(leBytes(val)) };
}

export function decodeSparseEdit(e: { idx: string; val: string }): { idx: Int32Array; val: Float32Array } {
  const ib = base64ToBytes(e.idx);
  const vb = base64ToBytes(e.val);
  const idx = viewOf(ib.buffer as ArrayBuffer, "int32", ib.byteOffset, ib.length / 4) as Int32Array;
  const val = viewOf(vb.buffer as ArrayBuffer, "float32", vb.byteOffset, vb.length / 4) as Float32Array;
  if (idx.length !== val.length) throw new RangeError("idx and val lengths differ");
  return { idx, val };
}

/** Dense per-row edit array → sparse (rows where the value is non-zero and finite). */
export function denseToSparse(dense: Float32Array): { idx: Int32Array; val: Float32Array } {
  let m = 0;
  for (let i = 0; i < dense.length; i++) if (dense[i] !== 0 && Number.isFinite(dense[i])) m++;
  const idx = new Int32Array(m);
  const val = new Float32Array(m);
  let k = 0;
  for (let i = 0; i < dense.length; i++) {
    const v = dense[i];
    if (v !== 0 && Number.isFinite(v)) {
      idx[k] = i;
      val[k] = v;
      k++;
    }
  }
  return { idx, val };
}

/**
 * Raw body for `PUT /api/runs/{rid}/blobs?kind=edit`: Int32 idx[m] then Float32 val[m],
 * sent with `X-SPARC-Count: m`.
 */
export function editBlobBody(idx: Int32Array, val: Float32Array): { body: Uint8Array; count: number } {
  if (idx.length !== val.length) throw new RangeError("idx and val lengths differ");
  const m = idx.length;
  const out = new Uint8Array(m * 8);
  out.set(leBytes(idx), 0);
  out.set(leBytes(val), m * 4);
  return { body: out, count: m };
}
