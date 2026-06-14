import { describe, expect, it } from "vitest";
import { shouldUseMockAnalysis } from "./analysisMode";

describe("shouldUseMockAnalysis", () => {
  const file = new File(["x"], "sample.npy", { type: "application/octet-stream" });

  it("uses mock for explicit simulation button", () => {
    expect(shouldUseMockAnalysis(file, true)).toBe(true);
  });

  it("uses mock when no file (loadDemo path)", () => {
    expect(shouldUseMockAnalysis(null, true)).toBe(true);
    expect(shouldUseMockAnalysis(null, false)).toBe(true);
  });

  it("uses live backend when file present and not simulation", () => {
    expect(shouldUseMockAnalysis(file, false)).toBe(false);
  });
});
