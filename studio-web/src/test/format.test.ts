import { describe, expect, it } from "vitest";
import {
  MINUS,
  coolerWarmer,
  fmtBytes,
  fmtBytesOf,
  fmtClock,
  fmtCount,
  fmtDuration,
  fmtDurationRange,
  fmtEta,
  fmtInt,
  fmtNum,
  fmtPct,
  fmtRange,
  fmtRelative,
  fmtSig,
  fmtSigned,
  fmtSignedValue,
  fmtTempChange,
  fmtValue,
  unitLabel,
} from "../theme/format";

describe("Unicode minus", () => {
  it("is U+2212", () => expect(MINUS).toBe("−"));
  it("formats negatives with U+2212, never a hyphen", () => {
    expect(fmtNum(-1234.5, 1)).toBe("−1,234.5");
    expect(fmtNum(-0.41, 2)).toBe("−0.41");
    expect(fmtNum(-0.41, 2)).not.toContain("-");
    expect(fmtInt(-54701)).toBe("−54,701");
  });
  it("prints rounded-away negatives as zero", () => {
    expect(fmtNum(-0.0004, 2)).toBe("0.00");
    expect(fmtSigned(-0.0004, 2)).toBe("0.00");
  });
  it("signs explicitly", () => {
    expect(fmtSigned(0.41)).toBe("+0.41");
    expect(fmtSigned(-0.41)).toBe("−0.41");
    expect(fmtSignedValue(-0.41, "degF")).toBe("−0.41 °F");
  });
  it("keeps the minus in ranges and significant-digit output", () => {
    expect(fmtRange(-0.52, -0.3, "degF")).toBe("−0.52 to −0.30 °F");
    expect(fmtRange(0.3, 0.52, "degF")).toBe("0.30–0.52 °F");
    expect(fmtSig(-0.012345, 3)).toBe("−0.0123");
  });
  it("shows missing values as an em dash", () => {
    expect(fmtNum(null)).toBe("—");
    expect(fmtNum(NaN)).toBe("—");
    expect(fmtBytes(undefined)).toBe("—");
  });
});

describe("units", () => {
  it("maps unit codes", () => {
    expect(unitLabel("degF")).toBe("°F");
    expect(unitLabel("pct")).toBe("%");
    expect(unitLabel("km2")).toBe("km²");
    expect(unitLabel("pp")).toBe("pp");
    expect(unitLabel(null)).toBe("");
  });
  it("attaches units", () => {
    expect(fmtValue(0.41, "degF")).toBe("0.41 °F");
    expect(fmtValue(12, "%", 0)).toBe("12%");
    expect(fmtPct(0.123)).toBe("12%");
    expect(fmtPct(0.9, 1)).toBe("90.0%");
  });
  it("words temperature changes cooler/warmer", () => {
    expect(fmtTempChange(-0.41, "degF")).toBe("0.41 °F cooler");
    expect(fmtTempChange(0.2, "degF")).toBe("0.20 °F warmer");
    expect(fmtTempChange(0.001, "degF")).toBe("no change");
    expect(coolerWarmer(-1)).toBe("cooler");
  });
  it("counts with plurals", () => {
    expect(fmtCount(1, "cell")).toBe("1 cell");
    expect(fmtCount(1284, "cell")).toBe("1,284 cells");
  });
});

describe("durations", () => {
  it("formats seconds, minutes, hours and days", () => {
    expect(fmtDuration(0.3)).toBe("0.3 s");
    expect(fmtDuration(4)).toBe("4 s");
    expect(fmtDuration(12.4)).toBe("12 s");
    expect(fmtDuration(252)).toBe("4 min 12 s");
    expect(fmtDuration(240)).toBe("4 min");
    expect(fmtDuration(26 * 60 + 10)).toBe("26 min");
    expect(fmtDuration(100 * 60)).toBe("1 h 40 m");
    expect(fmtDuration(3600)).toBe("1 h");
    expect(fmtDuration(3599)).toBe("1 h");
    expect(fmtDuration(2 * 86400 + 3 * 3600)).toBe("2 d 3 h");
    expect(fmtDuration(-90)).toBe("−1 min 30 s");
  });
  it("formats ranges", () => {
    expect(fmtDurationRange(240, 360)).toBe("4–6 min");
    expect(fmtDurationRange(40, 60)).toBe("40 s–1 min");
    expect(fmtDurationRange(40, 55)).toBe("40–55 s");
    // SPEC §2 J3: "estimated at 1 h 40 m–1 h 55 m"
    expect(fmtDurationRange(100 * 60, 115 * 60)).toBe("1 h 40 m–1 h 55 m");
    expect(fmtDurationRange(7200, 9000)).toBe("2 h–2 h 30 m");
  });
  it("formats ETAs and clocks", () => {
    expect(fmtEta(720, 600, 900)).toBe("≈12 min (10–15 min)");
    expect(fmtEta(null, 240, 360)).toBe("≈4–6 min");
    expect(fmtClock(75)).toBe("1:15");
    expect(fmtClock(3725)).toBe("1:02:05");
  });
  it("formats relative times", () => {
    const now = Date.parse("2026-10-01T12:00:00Z");
    expect(fmtRelative("2026-10-01T11:59:40Z", now)).toBe("just now");
    expect(fmtRelative("2026-10-01T11:57:00Z", now)).toBe("3 min ago");
    expect(fmtRelative("2026-10-01T10:00:00Z", now)).toBe("2 h ago");
  });
});

describe("bytes", () => {
  it("uses decimal units", () => {
    expect(fmtBytes(512)).toBe("512 B");
    expect(fmtBytes(218_804)).toBe("219 kB");
    expect(fmtBytes(525e6)).toBe("525 MB");
    expect(fmtBytes(1.2e9)).toBe("1.2 GB");
    expect(fmtBytes(999_600)).toBe("1.0 MB");
    expect(fmtBytes(-2048)).toBe("−2.0 kB");
  });
  it("formats transfer progress", () => {
    expect(fmtBytesOf(312e6, 525e6)).toBe("312/525 MB");
  });
});
