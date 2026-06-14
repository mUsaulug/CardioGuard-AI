import { z } from "zod";

const recordOfNumbers = z.record(z.string(), z.number());

export const SuperclassApiResponseSchema = z.object({
  mode: z.string(),
  probabilities: recordOfNumbers,
  predicted_labels: z.array(z.string()),
  thresholds: recordOfNumbers,
  primary: z.object({
    label: z.string(),
    confidence: z.number(),
    rule: z.string(),
  }),
  sources: z.object({
    cnn: recordOfNumbers,
    xgb: recordOfNumbers.nullable().optional(),
    ensemble: recordOfNumbers,
  }),
  versions: z.object({
    model_hash: z.string(),
    threshold_hash: z.string(),
    api_version: z.string(),
    timestamp: z.string(),
  }),
  xai: z
    .object({
      enabled: z.boolean(),
      run_id: z.string().nullable(),
      artifacts: z.array(
        z.object({
          type: z.string(),
          name: z.string(),
          url: z.string(),
          mime: z.string(),
        }),
      ),
    })
    .nullable()
    .optional(),
  consistency: z
    .object({
      agreement: z.string(),
      triage_level: z.string(),
      warnings: z.array(z.string()),
      superclass_mi_prob: z.number(),
      binary_mi_prob: z.number(),
      superclass_mi_decision: z.boolean().optional(),
      binary_mi_decision: z.boolean().optional(),
    })
    .nullable()
    .optional(),
  explanation: z
    .object({
      narrative: z.string(),
      coherence_score: z.number(),
      sanity_passed: z.boolean().nullable(),
      gradcam_summary: z.string(),
      shap_summary: z.string(),
      dominant_source: z.string(),
      conflicts: z.array(z.string()),
    })
    .nullable()
    .optional(),
  localization: z
    .object({
      mi_detected: z.boolean(),
      regions: z.array(z.string()),
      probabilities: recordOfNumbers,
      labels: z.array(z.string()),
      labels_tr: z.record(z.string(), z.string()),
    })
    .nullable()
    .optional(),
  glossary: z.record(z.string(), z.string()).optional(),
  airesult: z.record(z.string(), z.unknown()).nullable().optional(),
  latency_ms: z.number().nullable().optional(),
});

export type ParsedSuperclassApiResponse = z.infer<typeof SuperclassApiResponseSchema>;

export function parseSuperclassApiResponse(data: unknown): ParsedSuperclassApiResponse {
  return SuperclassApiResponseSchema.parse(data);
}
