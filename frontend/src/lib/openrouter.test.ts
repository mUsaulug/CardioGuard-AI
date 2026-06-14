import { describe, it, expect } from "vitest";
import {
  isClinicalAdviceRequest,
  looksLikeEmptyLlmRefusal,
  buildAdviceRefusalAnswer,
} from "./openrouter";
import type { AnalysisContext } from "./types";

function minimalCtx(): AnalysisContext {
  return {
    sessionId: "s1",
    fileName: "ecg.npz",
    timestamp: "2026-06-14T00:00:00Z",
    primary: { label: "MI", confidence: 0.4, rule: "r" },
    predictedLabels: ["MI"],
    probabilities: { MI: 0.4, STTC: 0.1, CD: 0.1, HYP: 0.05, NORM: 0.6 },
    thresholds: { MI: 0.16 },
    sources: { cnn: { MI: 0.5 }, xgb: { MI: 0.35 }, ensemble: { MI: 0.4 } },
    consistency: null,
    localization: null,
    xai: null,
    xaiArtifacts: [],
    runId: null,
    latencyMs: 880,
    glossary: {},
  };
}

describe("openrouter helpers", () => {
  it("detects clinical advice requests", () => {
    expect(isClinicalAdviceRequest("Hangi ilaç kullanmalıyım?")).toBe(true);
    expect(isClinicalAdviceRequest("STTC ne anlama geliyor?")).toBe(false);
  });

  it("detects empty LLM refusals", () => {
    expect(looksLikeEmptyLlmRefusal("Sorry, I cannot help with that.")).toBe(true);
    expect(looksLikeEmptyLlmRefusal("MI bölgesi anterior olabilir.")).toBe(false);
  });

  it("builds advice refusal with primary label", () => {
    const text = buildAdviceRefusalAnswer(minimalCtx());
    expect(text).toContain("Tedavi");
    expect(text).toContain("Miyokard");
  });
});
