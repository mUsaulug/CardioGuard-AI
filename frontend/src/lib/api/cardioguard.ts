import type { AnalyzeOptions } from "@/lib/types";
import { parseSuperclassApiResponse } from "@/lib/api/superclassSchema";

/** Max wait for ML inference (XAI can be slow). */
const INFERENCE_TIMEOUT_MS = 120_000;

export interface ExplanationInfo {
  narrative: string;
  coherence_score: number;
  sanity_passed: boolean | null;
  gradcam_summary: string;
  shap_summary: string;
  dominant_source: string;
  conflicts: string[];
}

export interface LocalizationInline {
  mi_detected: boolean;
  regions: string[];
  probabilities: Record<string, number>;
  labels: string[];
  labels_tr: Record<string, string>;
}

export interface SuperclassApiResponse {
  mode: string;
  probabilities: Record<string, number>;
  predicted_labels: string[];
  thresholds: Record<string, number>;
  primary: { label: string; confidence: number; rule: string };
  sources: {
    cnn: Record<string, number>;
    xgb: Record<string, number> | null;
    ensemble: Record<string, number>;
  };
  versions: {
    model_hash: string;
    threshold_hash: string;
    api_version: string;
    timestamp: string;
  };
  xai: {
    enabled: boolean;
    run_id: string | null;
    artifacts: Array<{ type: string; name: string; url: string; mime: string }>;
  } | null;
  consistency: {
    agreement: string;
    triage_level: string;
    warnings: string[];
    superclass_mi_prob: number;
    binary_mi_prob: number;
    superclass_mi_decision?: boolean;
    binary_mi_decision?: boolean;
  } | null;
  explanation: ExplanationInfo | null;
  localization: LocalizationInline | null;
  glossary: Record<string, string>;
  airesult?: Record<string, unknown> | null;
  latency_ms?: number | null;
}

export function normalizeBaseUrl(url: string): string {
  return url.replace(/\/+$/, "");
}

export async function predictSuperclass(
  file: File,
  options: AnalyzeOptions,
  backendUrl: string,
): Promise<SuperclassApiResponse> {
  const base = normalizeBaseUrl(backendUrl);
  // UI slider = XGB weight; backend ensemble_weight = CNN weight (w*cnn + (1-w)*xgb)
  const cnnWeight = 1 - options.ensemble;
  const params = new URLSearchParams({
    ensemble_weight: String(cnnWeight),
    explain: String(options.explain),
    sanity_check: String(options.sanityCheck),
    full: "true",
  });

  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${base}/predict/superclass?${params.toString()}`, {
    method: "POST",
    body: form,
    signal: AbortSignal.timeout(INFERENCE_TIMEOUT_MS),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const err = (await res.json()) as { detail?: string };
      if (err.detail) detail = err.detail;
    } catch {
      /* ignore */
    }
    throw new Error(`Backend hata (${res.status}): ${detail}`);
  }

  const data = await res.json();
  try {
    return parseSuperclassApiResponse(data) as SuperclassApiResponse;
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Geçersiz API yanıtı";
    throw new Error(`Backend yanıtı doğrulanamadı: ${detail}`);
  }
}

export interface BackendStatus {
  healthy: boolean;
  ready: boolean;
  degraded: boolean;
}

export async function fetchBackendStatus(backendUrl: string): Promise<BackendStatus> {
  const base = normalizeBaseUrl(backendUrl);
  try {
    const readyRes = await fetch(`${base}/ready`, {
      method: "GET",
      signal: AbortSignal.timeout(8000),
    });
    if (readyRes.ok) {
      const data = (await readyRes.json()) as { ready?: boolean; degraded?: boolean };
      return {
        healthy: true,
        ready: data.ready === true,
        degraded: Boolean(data.degraded),
      };
    }
  } catch {
    // fall through to /health
  }

  try {
    const healthRes = await fetch(`${base}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(8000),
    });
    return { healthy: healthRes.ok, ready: false, degraded: false };
  } catch {
    return { healthy: false, ready: false, degraded: false };
  }
}

export async function testBackendConnection(backendUrl: string): Promise<boolean> {
  const status = await fetchBackendStatus(backendUrl);
  return status.ready || status.healthy;
}
