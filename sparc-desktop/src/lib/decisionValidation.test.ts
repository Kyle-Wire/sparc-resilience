import { describe, it, expect } from "vitest";
import { parseBudget } from "./decisionValidation";

describe("parseBudget", () => {
  it("returns null for empty string", () => {
    expect(parseBudget("").value).toBeNull();
    expect(parseBudget("   ").value).toBeNull();
  });

  it("returns null error for empty string", () => {
    expect(parseBudget("").error).toBeNull();
  });

  it("rejects non-numeric input with an error message", () => {
    const result = parseBudget("abc");
    expect(result.value).toBeNull();
    expect(result.error).toMatch(/valid number/i);
  });

  it("rejects negative values with an error message", () => {
    const result = parseBudget("-100");
    expect(result.value).toBeNull();
    expect(result.error).toMatch(/positive/i);
  });

  it("rejects zero with an error message", () => {
    const result = parseBudget("0");
    expect(result.value).toBeNull();
    expect(result.error).toMatch(/positive/i);
  });

  it("rejects values above 1 trillion with an error message", () => {
    const result = parseBudget("1000000000001");
    expect(result.value).toBeNull();
    expect(result.error).toMatch(/too large/i);
  });

  it("accepts a valid positive number", () => {
    const result = parseBudget("50000");
    expect(result.value).toBe(50000);
    expect(result.error).toBeNull();
  });

  it("accepts decimal values", () => {
    const result = parseBudget("1234.56");
    expect(result.value).toBeCloseTo(1234.56);
    expect(result.error).toBeNull();
  });

  it("accepts values with surrounding whitespace", () => {
    const result = parseBudget("  9999  ");
    expect(result.value).toBe(9999);
    expect(result.error).toBeNull();
  });
});
