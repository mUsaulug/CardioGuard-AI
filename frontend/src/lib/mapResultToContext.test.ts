import { describe, it, expect } from "vitest";
import { mapResultToContext } from "./mapResultToContext";
import type { SuperclassApiResponse } from "./api/cardioguard";

function makeApi(overrides: Partial<SuperclassApiResponse> = {}): SuperclassApiResponse {
  return {
    mode: "multilabel-superclass",
    probabilities: { MI: 0.4, STTC: 0.2, CD: 0.1, HYP: 0.05, NORM: 0.6 },
    predicted_labels: ["MI"],
    thresholds: { MI: 0.16, STTC: 0.26, CD: 0.28, HYP: 0.19 },
    primary: { label: "MI", confidence: 0.4, rule: "MI-first-then-priority" },
    sources: {
      cnn: { MI: 0.69, STTC: 0.2, CD: 0.1, HYP: 0.05, NORM: 0.3 },
      xgb: { MI: 0.35, STTC: 0.2, CD: 0.1, HYP: 0.05, NORM: 0.65 },
      ensemble: { MI: 0.4, STTC: 0.2, CD: 0.1, HYP: 0.05, NORM: 0.6 },
    },
    versions: {
      model_hash: "abc",
      threshold_hash: "def",
      api_version: "1.2.0",
      timestamp: "2026-06-09T00:00:00Z",
    },
    xai: {
      enabled: true,
      run_id: "20260609_run__abc",
      artifacts: [
        { type: "report_png", name: "r.png", url: "/runs/20260609_run__abc/visuals/r.png", mime: "image/png" },
        { type: "narrative_md", name: "n.md", url: "/runs/20260609_run__abc/text/n.md", mime: "text/markdown" },
      ],
    },
    consistency: null,
    explanation: {
      narrative: "n",
      coherence_score: 0.85,
      sanity_passed: true,
      gradcam_summary: "g",
      shap_summary: "s",
      dominant_source: "XGBoost (Feature)",
      conflicts: [],
    },
    localization: null,
    glossary: { MI: "Miyokard enfarktüsü" },
    latency_ms: 880,
    ...overrides,
  };
}

describe("mapResultToContext", () => {
  it("maps xai artifacts to absolute URLs using the backend base", () => {
    const ctx = mapResultToContext(makeApi(), "ecg.npz", "http://localhost:8000/");
    expect(ctx.xaiArtifacts).toHaveLength(2);
    expect(ctx.xaiArtifacts[0].url).toBe(
      "http://localhost:8000/runs/20260609_run__abc/visuals/r.png",
    );
    expect(ctx.xaiArtifacts[0].mime).toBe("image/png");
  });

  it("carries run_id and latency", () => {
    const ctx = mapResultToContext(makeApi(), "ecg.npz", "http://localhost:8000");
    expect(ctx.runId).toBe("20260609_run__abc");
    expect(ctx.latencyMs).toBe(880);
  });

  it("does not double-prefix already-absolute artifact URLs", () => {
    const api = makeApi({
      xai: {
        enabled: true,
        run_id: "r",
        artifacts: [
          { type: "report_png", name: "r.png", url: "https://cdn.example/r.png", mime: "image/png" },
        ],
      },
    });
    const ctx = mapResultToContext(api, "ecg.npz", "http://localhost:8000");
    expect(ctx.xaiArtifacts[0].url).toBe("https://cdn.example/r.png");
  });

  it("defaults artifacts/runId/latency safely when xai is absent", () => {
    const ctx = mapResultToContext(makeApi({ xai: null, latency_ms: null }), "ecg.npz", "http://x");
    expect(ctx.xaiArtifacts).toEqual([]);
    expect(ctx.runId).toBeNull();
    expect(ctx.latencyMs).toBeNull();
  });

  it("maps API versions metadata", () => {
    const ctx = mapResultToContext(makeApi(), "ecg.npz", "http://localhost:8000");
    expect(ctx.versions?.model_hash).toBe("abc");
    expect(ctx.versions?.api_version).toBe("1.2.0");
    expect(ctx.versions?.threshold_hash).toBe("def");
  });

  it("preserves cnn/xgb/ensemble source divergence (primary uses ensemble)", () => {
    const ctx = mapResultToContext(makeApi(), "ecg.npz", "http://x");
    expect(ctx.primary.confidence).toBe(0.4); // ensemble
    expect(ctx.sources.cnn.MI).toBe(0.69); // raw CNN differs
  });
});
