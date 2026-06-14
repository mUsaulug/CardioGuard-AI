export type PathologyKey = "MI" | "STTC" | "CD" | "HYP" | "NORM";

export interface AnalysisContext {
  sessionId: string;
  fileName: string;
  timestamp: string;
  primary: { label: string; confidence: number; rule: string };
  predictedLabels: string[];
  probabilities: Record<PathologyKey, number>;
  thresholds: Partial<Record<PathologyKey, number>>;
  sources: {
    cnn: Record<string, number>;
    xgb: Record<string, number>;
    ensemble: Record<string, number>;
  };
  consistency: {
    agreement: string;
    triage_level: string;
    warnings: string[];
    superclass_mi_prob: number;
    binary_mi_prob: number;
    superclass_mi_decision?: boolean;
    binary_mi_decision?: boolean;
  } | null;
  localization: {
    regions: string[];
    probabilities: Record<string, number>;
    labels_tr: Record<string, string>;
  } | null;
  xai: {
    narrative: string;
    coherence_score: number;
    sanity_passed: boolean | null;
    gradcam_summary: string;
    shap_summary: string;
  } | null;
  /** XAI artifact files served by the backend, with absolute URLs. */
  xaiArtifacts: XaiArtifact[];
  /** Backend XAI run identifier (when explain=true). */
  runId: string | null;
  /** Server-side inference latency in ms (null for mock/demo). */
  latencyMs: number | null;
  glossary: Record<string, string>;
  /** API version metadata from backend (when available). */
  versions?: {
    model_hash: string;
    threshold_hash: string;
    api_version: string;
  };
}

export interface XaiArtifact {
  type: string;
  name: string;
  url: string;
  mime: string;
}

export type ChatRole = "user" | "assistant";

export type MessageSource = "llm" | "template" | "auto" | "rule";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  pending?: boolean;
  /** How an assistant message was produced. Undefined for user messages. */
  source?: MessageSource;
}

export type AppState = "welcome" | "analyzing" | "results";

export type LlmStatus = "active" | "limit" | "offline" | "error";

export interface StoredSession {
  context: AnalysisContext;
  messages: ChatMessage[];
  isDemo: boolean;
  timestamp: number;
  /** Bumped when AnalysisContext shape changes; old sessions are discarded. */
  schema?: number;
}

export interface AnalyzeOptions {
  explain: boolean;
  sanityCheck: boolean;
  ensemble: number;
}
