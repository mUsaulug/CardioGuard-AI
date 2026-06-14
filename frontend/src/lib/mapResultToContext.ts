import { normalizeBaseUrl } from "@/lib/api/cardioguard";
import type { SuperclassApiResponse } from "@/lib/api/cardioguard";
import type { AnalysisContext, PathologyKey, XaiArtifact } from "@/lib/types";

export function mapResultToContext(
  api: SuperclassApiResponse,
  fileName: string,
  backendUrl: string,
  sessionId?: string,
): AnalysisContext {
  const xai = api.explanation
    ? {
        narrative: api.explanation.narrative,
        coherence_score: api.explanation.coherence_score,
        sanity_passed: api.explanation.sanity_passed,
        gradcam_summary: api.explanation.gradcam_summary,
        shap_summary: api.explanation.shap_summary,
      }
    : null;

  const localization = api.localization
    ? {
        regions: api.localization.regions,
        probabilities: api.localization.probabilities,
        labels_tr: api.localization.labels_tr,
      }
    : null;

  // Backend returns relative artifact URLs (/runs/{id}/...). Make them absolute.
  const base = normalizeBaseUrl(backendUrl);
  const xaiArtifacts: XaiArtifact[] = (api.xai?.artifacts ?? []).map((a) => ({
    type: a.type,
    name: a.name,
    mime: a.mime,
    url: /^https?:\/\//.test(a.url) ? a.url : `${base}${a.url.startsWith("/") ? "" : "/"}${a.url}`,
  }));

  return {
    sessionId: sessionId ?? `live-${Date.now().toString(36)}`,
    fileName,
    timestamp: api.versions.timestamp || new Date().toISOString(),
    primary: api.primary,
    predictedLabels: api.predicted_labels,
    probabilities: api.probabilities as Record<PathologyKey, number>,
    thresholds: api.thresholds,
    sources: {
      cnn: api.sources.cnn,
      xgb: api.sources.xgb ?? {},
      ensemble: api.sources.ensemble,
    },
    consistency: api.consistency,
    localization,
    xai,
    xaiArtifacts,
    runId: api.xai?.run_id ?? null,
    latencyMs: api.latency_ms ?? null,
    glossary: api.glossary ?? {},
    versions: {
      model_hash: api.versions.model_hash,
      threshold_hash: api.versions.threshold_hash,
      api_version: api.versions.api_version,
    },
  };
}
