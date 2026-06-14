import { describe, it, expect } from "vitest";
import { SuperclassApiResponseSchema, parseSuperclassApiResponse } from "./superclassSchema";

const validFixture = {
  mode: "multilabel-superclass",
  probabilities: { MI: 0.4, STTC: 0.2, CD: 0.1, HYP: 0.05, NORM: 0.6 },
  predicted_labels: ["MI"],
  thresholds: { MI: 0.16 },
  primary: { label: "MI", confidence: 0.4, rule: "MI-first" },
  sources: { cnn: { MI: 0.5 }, xgb: { MI: 0.35 }, ensemble: { MI: 0.4 } },
  versions: {
    model_hash: "abc",
    threshold_hash: "def",
    api_version: "1.2.0",
    timestamp: "2026-06-14T00:00:00Z",
  },
};

describe("SuperclassApiResponseSchema", () => {
  it("accepts a valid API payload", () => {
    const parsed = parseSuperclassApiResponse(validFixture);
    expect(parsed.primary.label).toBe("MI");
  });

  it("rejects missing primary", () => {
    const { primary: _, ...rest } = validFixture;
    expect(() => SuperclassApiResponseSchema.parse(rest)).toThrow();
  });

  it("rejects non-numeric probability", () => {
    expect(() =>
      SuperclassApiResponseSchema.parse({
        ...validFixture,
        probabilities: { MI: "bad" },
      }),
    ).toThrow();
  });
});
