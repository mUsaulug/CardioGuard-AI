import type { AnalyzeOptions } from "@/lib/types";

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

  return res.json() as Promise<SuperclassApiResponse>;
}

export async function testBackendConnection(backendUrl: string): Promise<boolean> {
  try {
    const res = await fetch(`${normalizeBaseUrl(backendUrl)}/health`, {
      method: "GET",
    });
    return res.ok;
  } catch {
    return false;
  }
}
