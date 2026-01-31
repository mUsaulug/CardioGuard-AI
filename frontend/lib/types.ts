export interface HealthResponse {
  status: string;
  timestamp: string;
}

export interface ReadyResponse {
  ready: boolean;
  models_loaded: {
    superclass: boolean;
    localization: boolean;
    xgb: boolean;
    thresholds: boolean;
  };
  message: string;
}

export interface Artifact {
  type: string;
  name: string;
  url: string;
  mime: string;
}

export interface XaiSchema {
  enabled: boolean;
  run_id: string | null;
  run_dir: string | null;
  artifacts: Artifact[];
  highlights: object[] | null;
  sanity: object | null;
}

export interface Versions {
  model_hash: string;
  threshold_hash: string;
  api_version: string;
  timestamp: string;
}

// --- Consistency Guard Types ---

export interface ConsistencyInfo {
  agreement: string;
  triage_level: string;
  superclass_mi_prob: number;
  binary_mi_prob: number;
  superclass_mi_decision: boolean;
  binary_mi_decision: boolean;
  warnings: string[];
}

// --- Superclass Types ---

export interface SuperclassProbabilities {
  MI: number;
  STTC: number;
  CD: number;
  HYP: number;
  NORM: number;
}

export interface SuperclassResponse {
  mode: string;
  probabilities: SuperclassProbabilities;
  predicted_labels: string[];
  thresholds: {
    MI: number;
    STTC: number;
    CD: number;
    HYP: number;
  };
  primary: {
    label: string;
    confidence: number;
    rule: string;
  };
  sources: {
    cnn: SuperclassProbabilities;
    xgb: SuperclassProbabilities | null;
    ensemble: SuperclassProbabilities;
  };
  versions: Versions;
  xai: XaiSchema | null;
  consistency: ConsistencyInfo | null;
}

// --- MI Localization Types ---

export interface LocalizationProbabilities {
  AMI: number;
  ASMI: number;
  ALMI: number;
  IMI: number;
  LMI: number;
}

export interface LocalizationResponse {
  mi_detected: boolean;
  regions: string[];
  probabilities: LocalizationProbabilities;
  label_space: string;
  labels: string[];
  mapping_source: string;
  mapping_fingerprint: string;
  localization_head_type: string;
  xai: XaiSchema | null;
}

export interface ApiError {
  error: string;
  detail?: string;
}
