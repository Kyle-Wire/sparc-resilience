import { describe, expect, it } from "vitest";
import {
  asFloat32,
  asInt64,
  base64ToBytes,
  bytesToBase64,
  countBits,
  decodeBitset,
  decodeSparseEdit,
  denseToSparse,
  editBlobBody,
  encodeBitset,
  encodeSparseEdit,
  int64ToNumbers,
  pack,
  packBits,
  parseOffsets,
  unpack,
  unpackBits,
  viewOf,
} from "../api/binary";

describe("bitset codec (LSB-first, base64)", () => {
  it("packs row 8i+k into bit k of byte i", () => {
    const mask = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1];
    expect([...packBits(mask)]).toEqual([0b00000001, 0b00000010]);
    expect([...packBits([0, 0, 0, 0, 0, 0, 0, 1])]).toEqual([0x80]);
  });
  it("matches a hand-computed wire value", () => {
    // rows 0, 2, 9 selected out of 12 → bytes [0b101, 0b10] → base64 "BQI="
    const mask = new Uint8Array(12);
    mask[0] = mask[2] = mask[9] = 1;
    expect(encodeBitset(mask)).toBe("BQI=");
    expect([...decodeBitset("BQI=", 12)]).toEqual([...mask]);
  });
  it("round-trips random masks of awkward lengths", () => {
    for (const n of [1, 7, 8, 9, 54701]) {
      const m = Uint8Array.from({ length: n }, (_, i) => ((i * 2654435761) >>> 0) % 3 === 0 ? 1 : 0);
      const b = encodeBitset(m);
      expect(decodeBitset(b, n)).toEqual(m);
      expect(countBits(decodeBitset(b, n))).toBe(countBits(m));
    }
  });
  it("rejects a bitset that is too short", () => {
    expect(() => unpackBits(new Uint8Array(1), 9)).toThrow(RangeError);
  });
});

describe("base64", () => {
  it("round-trips and accepts URL-safe input", () => {
    const b = Uint8Array.from([0, 255, 128, 62, 63, 250]);
    const s = bytesToBase64(b);
    expect(base64ToBytes(s)).toEqual(b);
    expect(base64ToBytes(s.replace(/\+/g, "-").replace(/\//g, "_"))).toEqual(b);
  });
});

describe("typed views and X-SPARC-Offsets", () => {
  it("views little-endian Float32 and Int64", () => {
    const buf = new ArrayBuffer(8);
    new DataView(buf).setFloat32(0, 1.5, true);
    new DataView(buf).setFloat32(4, NaN, true);
    const f = asFloat32(buf);
    expect(f[0]).toBe(1.5);
    expect(Number.isNaN(f[1])).toBe(true);
    const ib = new ArrayBuffer(16);
    new DataView(ib).setBigInt64(0, 9007199254740991n, true);
    new DataView(ib).setBigInt64(8, -5n, true);
    expect(int64ToNumbers(asInt64(ib))).toEqual([9007199254740991, -5]);
  });
  it("copies unaligned views instead of throwing", () => {
    const buf = new ArrayBuffer(12);
    new DataView(buf).setInt32(2, 77, true);
    expect((viewOf(buf, "int32", 2, 1) as Int32Array)[0]).toBe(77);
  });
  it("parses offsets and unpacks grid.bin", () => {
    const n = 3;
    const { buffer, offsets } = pack([
      { name: "ix", dtype: "int32", data: Int32Array.from([0, 1, 2]) },
      { name: "iy", dtype: "int32", data: Int32Array.from([2, 1, 0]) },
      { name: "lon", dtype: "float32", data: Float32Array.from([-71.4, -71.3, -71.2]) },
      { name: "lat", dtype: "float32", data: Float32Array.from([41.8, 41.81, 41.82]) },
      { name: "zone", dtype: "int16", data: Int16Array.from([1, 2, 3]) },
    ]);
    for (const o of offsets) expect(o.offset % 8).toBe(0);
    const parsed = parseOffsets(JSON.stringify(offsets));
    expect(parsed).toEqual(offsets);
    const a = unpack(buffer, parsed);
    expect([...(a.ix as Int32Array)]).toEqual([0, 1, 2]);
    expect([...(a.iy as Int32Array)]).toEqual([2, 1, 0]);
    expect((a.lon as Float32Array)[0]).toBeCloseTo(-71.4, 5);
    expect([...(a.zone as Int16Array)]).toEqual([1, 2, 3]);
    expect(a.ix.length).toBe(n);
  });
  it("rejects malformed offsets", () => {
    expect(() => parseOffsets("{}")).toThrow();
    expect(() => parseOffsets('[{"name":"x","dtype":"float64","offset":0,"length":1}]')).toThrow();
    expect(() => parseOffsets("not json")).toThrow();
  });
});

describe("sparse edits", () => {
  it("encodes {idx, val} and the raw edit blob", () => {
    const dense = new Float32Array(6);
    dense[1] = 10;
    dense[4] = -2.5;
    const { idx, val } = denseToSparse(dense);
    expect([...idx]).toEqual([1, 4]);
    expect([...val]).toEqual([10, -2.5]);
    const e = encodeSparseEdit(idx, val);
    const back = decodeSparseEdit(e);
    expect([...back.idx]).toEqual([1, 4]);
    expect([...back.val]).toEqual([10, -2.5]);
    const { body, count } = editBlobBody(idx, val);
    expect(count).toBe(2);
    const dv = new DataView(body.buffer);
    expect(dv.getInt32(0, true)).toBe(1);
    expect(dv.getInt32(4, true)).toBe(4);
    expect(dv.getFloat32(8, true)).toBe(10);
    expect(dv.getFloat32(12, true)).toBe(-2.5);
  });
});
