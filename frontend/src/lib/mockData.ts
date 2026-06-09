import type { AnalysisContext } from "./types";
import { GLOSSARY, LOCALIZATION_LABELS_TR } from "./glossary";

export function createMockContext(fileName = "ornek_ekg_record_00021.npy"): AnalysisContext {
  return {
    sessionId: `demo-${Date.now().toString(36)}`,
    fileName,
    timestamp: new Date().toISOString(),
    primary: { label: "MI", confidence: 0.873, rule: "MI-first-then-priority" },
    predictedLabels: ["MI", "STTC"],
    probabilities: { MI: 0.873, STTC: 0.452, CD: 0.118, HYP: 0.082, NORM: 0.127 },
    thresholds: { MI: 0.16, STTC: 0.256, CD: 0.276, HYP: 0.19 },
    sources: {
      cnn: { MI: 0.861, STTC: 0.471, CD: 0.131, HYP: 0.075, NORM: 0.119 },
      xgb: { MI: 0.885, STTC: 0.433, CD: 0.105, HYP: 0.089, NORM: 0.135 },
      ensemble: { MI: 0.873, STTC: 0.452, CD: 0.118, HYP: 0.082, NORM: 0.127 },
    },
    consistency: {
      agreement: "AGREE_MI",
      triage_level: "HIGH",
      superclass_mi_prob: 0.873,
      binary_mi_prob: 0.891,
      superclass_mi_decision: true,
      binary_mi_decision: true,
      warnings: [],
    },
    localization: {
      regions: ["ASMI"],
      probabilities: { AMI: 0.34, ASMI: 0.78, ALMI: 0.21, IMI: 0.12, LMI: 0.08 },
      labels_tr: LOCALIZATION_LABELS_TR,
    },
    xai: {
      narrative:
        "**V1-V4 derivasyonlarında** ST segmentinde belirgin aktivasyon gözlenmektedir. SHAP analizi, embedding boyutu 12-15 arasında yüksek katkı göstermektedir. Bu bulgular **anteroseptal** bölge ile uyumludur.\n\n- Grad-CAM odak bölgesi: V2-V4\n- SHAP baskın özellik: CNN embedding 12-15\n- Tutarlılık: Görsel ve istatistiksel kanıtlar uyumlu",
      coherence_score: 0.82,
      sanity_passed: true,
      gradcam_summary: "Anteroseptal derivasyonlarda (V2-V4) ST segment odaklı ısı haritası",
      shap_summary: "CNN embedding özellikleri MI sınıfını destekliyor",
    },
    xaiArtifacts: [],
    runId: null,
    latencyMs: null,
    glossary: GLOSSARY,
  };
}

export function buildAutoSummary(ctx: AnalysisContext): string {
  const pct = (n: number) => `%${(n * 100).toFixed(1)}`;
  const primaryTr = ctx.glossary[ctx.primary.label]?.split("—")[0]?.trim() || ctx.primary.label;
  const others = ctx.predictedLabels.filter((l) => l !== ctx.primary.label);
  const locLine =
    ctx.localization && ctx.localization.regions.length
      ? `\n\n**Lokalizasyon:** ${ctx.localization.regions
          .map((r) => `${ctx.localization!.labels_tr[r] || r}`)
          .join(", ")} bölgesi öne çıkmaktadır.`
      : "";
  const consLine = ctx.consistency
    ? `\n\n**Model uyumu:** Superclass ve binary MI modelleri ${
        ctx.consistency.agreement.startsWith("AGREE") ? "uyumlu" : "kısmen uyumlu"
      } (${triageTr(ctx.consistency.triage_level)} güven).`
    : "";
  const xaiLine = ctx.xai
    ? `\n\n**Açıklama güvenilirliği:** Görsel (Grad-CAM) ve istatistiksel (SHAP) kanıtlar uyumlu (coherence: ${pct(
        ctx.xai.coherence_score
      )}).`
    : "";

  return `## 📋 Klinik Özet

Bu EKG analizinde birincil bulgu **${primaryTr} (${ctx.primary.label})** olarak değerlendirilmiştir (güven: ${pct(
    ctx.primary.confidence
  )}).${
    others.length
      ? ` Ayrıca ${others.map((o) => o).join(", ")} eşik üzerinde tespit edilmiştir.`
      : ""
  }${locLine}${consLine}${xaiLine}

> ⚠️ Bu sistem tanı koymaz; klinik karar destek aracıdır. Nihai değerlendirme hekim tarafından yapılmalıdır.

Size nasıl yardımcı olabilirim? Örneğin:
- "MI ne demek?"
- "Neden STTC de yüksek?"
- "ASMI hangi derivasyonlarda görülür?"
- "Bu sonuç ne kadar güvenilir?"`;
}

function triageTr(level: string): string {
  const m: Record<string, string> = {
    HIGH: "YÜKSEK",
    MEDIUM: "ORTA",
    LOW: "DÜŞÜK",
    REVIEW: "İNCELEME",
  };
  return m[level] || level;
}
