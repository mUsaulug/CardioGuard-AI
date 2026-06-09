import type { AnalysisContext, ChatMessage } from "./types";
import { DISCLAIMER, PATHOLOGY_LABELS_TR } from "./glossary";
import { debugLog } from "./sessionDebugLog";

const ENDPOINT = "https://openrouter.ai/api/v1/chat/completions";

/**
 * Ücretsiz modeller — tek API key ile hepsi kullanılabilir.
 * Sıra: en dayanıklı / en hızlı önce. openrouter/free müsait free model seçer.
 */
export const FREE_MODEL_CHAIN = [
  "openrouter/free",
  "google/gemma-4-31b-it:free",
  "google/gemma-4-26b-a4b-it:free",
  "qwen/qwen3-next-80b-a3b-instruct:free",
  "meta-llama/llama-3.3-70b-instruct:free",
] as const;

export const DEFAULT_MODEL = FREE_MODEL_CHAIN[0];

const MAX_HISTORY_MESSAGES = 6;
const MAX_TOKENS_STREAM = 700;
const MAX_TOKENS_SYNC = 600;
/** Tek modelde bu süreden uzun beklenmez — sonraki modele geçilir. */
const PER_MODEL_TIMEOUT_MS = 22_000;

export class RateLimitError extends Error {}

export class OpenRouterError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "OpenRouterError";
    this.status = status;
  }
}

/** Geçici hatalarda bir sonraki ücretsiz modele geç. */
function isRetryableStatus(status: number): boolean {
  return status === 429 || status === 502 || status === 503 || status === 504;
}

function freeModelChain(): string[] {
  return [...FREE_MODEL_CHAIN];
}

function openRouterHeaders(apiKey: string): Record<string, string> {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${apiKey}`,
    "HTTP-Referer": typeof window !== "undefined" ? window.location.origin : "",
    "X-Title": "CardioGuard-AI",
  };
}

function createModelAbort(parent?: AbortSignal): { signal: AbortSignal; cleanup: () => void } {
  const ctrl = new AbortController();
  const timer = setTimeout(() => {
    ctrl.abort(new Error(`Model zaman aşımı (${PER_MODEL_TIMEOUT_MS / 1000}s)`));
  }, PER_MODEL_TIMEOUT_MS);

  const onParentAbort = () => ctrl.abort(parent?.reason);
  if (parent) {
    if (parent.aborted) onParentAbort();
    else parent.addEventListener("abort", onParentAbort, { once: true });
  }

  return {
    signal: ctrl.signal,
    cleanup: () => {
      clearTimeout(timer);
      parent?.removeEventListener("abort", onParentAbort);
    },
  };
}

function isTimeoutError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  if (err.name === "TimeoutError" || err.name === "AbortError") {
    return err.message.includes("zaman aşımı") || err.message.includes("timeout");
  }
  return err.message.includes("zaman aşımı");
}

async function readErrorSnippet(res: Response): Promise<string> {
  try {
    const text = await res.text();
    const parsed = JSON.parse(text) as { error?: { message?: string } };
    return parsed.error?.message || text.slice(0, 200);
  } catch {
    return res.statusText;
  }
}

/** Compact analysis payload — full ctx JSON blows free-tier limits. */
export function buildCompactContext(ctx: AnalysisContext): Record<string, unknown> {
  return {
    fileName: ctx.fileName,
    primary: ctx.primary,
    predictedLabels: ctx.predictedLabels,
    probabilities: ctx.probabilities,
    thresholds: ctx.thresholds,
    sources: ctx.sources,
    consistency: ctx.consistency,
    localization: ctx.localization,
    xai: ctx.xai
      ? {
          coherence_score: ctx.xai.coherence_score,
          gradcam_summary: ctx.xai.gradcam_summary,
          shap_summary: ctx.xai.shap_summary,
        }
      : null,
    glossary: ctx.glossary,
  };
}

export function buildSystemPrompt(ctx: AnalysisContext): string {
  return `Sen CardioGuard-AI klinik asistanısın. Türkçe yanıt ver.

GÖREV: EKG analiz sonuçlarını anlaşılır ve yapılandırılmış biçimde açıkla (özet + maddeler + kısa yorum).
KURALLAR:
- Sadece ANALIZ_VERISI kullan; internetten araştırma yapma, uydurma bilgi ekleme.
- Tanı koyma, tedavi/ilaç/yaşam tarzı tavsiyesi verme.
- Kullanıcı tavsiye, öneri, "ne yapmalı", internetten araştır isterse: kibarca reddet AMA aynı yanıtta
  bu oturumdaki bulguları (birincil sınıf, olasılıklar, lokalizasyon, Consistency Guard) özetle;
  "kesin karar için hekim değerlendirmesi gerekir" de. Boş "yardımcı olamam" deme.
- Her yanıt sonunda: _Bu bilgi karar destek amaçlıdır._

ANALIZ_VERISI: ${JSON.stringify(buildCompactContext(ctx))}`;
}

function buildMessages(
  ctx: AnalysisContext,
  userMessage: string,
  history: ChatMessage[],
): { role: string; content: string }[] {
  return [
    { role: "system", content: buildSystemPrompt(ctx) },
    ...history
      .filter((m) => !m.pending)
      .slice(-MAX_HISTORY_MESSAGES)
      .map((m) => ({ role: m.role, content: m.content })),
    { role: "user", content: userMessage },
  ];
}

async function readStreamBody(
  res: Response,
  onToken: (chunk: string) => void,
): Promise<string> {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("Yanıt akışı okunamadı");
  const decoder = new TextDecoder();
  let buffer = "";
  let acc = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const data = trimmed.slice(5).trim();
      if (data === "[DONE]") return acc;
      try {
        const json = JSON.parse(data);
        const token = json.choices?.[0]?.delta?.content;
        if (token) {
          acc += token;
          onToken(token);
        }
      } catch {
        /* partial SSE chunk */
      }
    }
  }
  return acc;
}

async function syncCompletion(
  apiKey: string,
  model: string,
  messages: { role: string; content: string }[],
  signal?: AbortSignal,
): Promise<string> {
  const res = await fetch(ENDPOINT, {
    method: "POST",
    signal,
    headers: openRouterHeaders(apiKey),
    body: JSON.stringify({
      model,
      messages,
      max_tokens: MAX_TOKENS_SYNC,
      temperature: 0.3,
      stream: false,
    }),
  });
  if (!res.ok) {
    const detail = await readErrorSnippet(res);
    throw new OpenRouterError(res.status, detail);
  }
  const json = (await res.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  return json.choices?.[0]?.message?.content?.trim() || "";
}

interface StreamCallbacks {
  onToken: (chunk: string) => void;
  signal?: AbortSignal;
  onProgress?: (message: string) => void;
}

export async function streamChat(
  apiKey: string,
  userMessage: string,
  ctx: AnalysisContext,
  history: ChatMessage[],
  cb: StreamCallbacks,
): Promise<void> {
  const messages = buildMessages(ctx, userMessage, history);
  const chain = freeModelChain();

  let lastStatus = 0;
  let lastDetail = "";
  let saw429 = false;

  debugLog("llm", "info", "LLM isteği başladı", {
    models: chain.length,
    userPreview: userMessage.slice(0, 80),
  });

  for (let i = 0; i < chain.length; i++) {
    const model = chain[i];
    const step = `${i + 1}/${chain.length}`;
    const progress = `Ücretsiz model deneniyor (${step}: ${model})…`;
    cb.onProgress?.(progress);
    debugLog("llm", "info", "Model deneniyor", { model, step });

    const { signal, cleanup } = createModelAbort(cb.signal);
    const t0 = performance.now();

    try {
      const res = await fetch(ENDPOINT, {
        method: "POST",
        signal,
        headers: openRouterHeaders(apiKey),
        body: JSON.stringify({
          model,
          messages,
          max_tokens: MAX_TOKENS_STREAM,
          temperature: 0.3,
          stream: true,
        }),
      });

      if (!res.ok) {
        lastStatus = res.status;
        lastDetail = `[${model}] ${await readErrorSnippet(res)}`;
        debugLog("llm", "warn", "Model HTTP hatası", {
          model,
          status: res.status,
          detail: lastDetail,
          ms: Math.round(performance.now() - t0),
        });
        if (res.status === 429) saw429 = true;
        if (isRetryableStatus(res.status)) continue;
        throw new OpenRouterError(res.status, lastDetail);
      }

      const streamed = await readStreamBody(res, cb.onToken);
      if (streamed.trim()) {
        debugLog("llm", "info", "LLM stream başarılı", {
          model,
          ms: Math.round(performance.now() - t0),
          chars: streamed.length,
        });
        return;
      }

      cb.onProgress?.(`Akış boş — ${model} sync deneniyor…`);
      const synced = await syncCompletion(apiKey, model, messages, signal);
      if (synced) {
        cb.onToken(synced);
        debugLog("llm", "info", "LLM sync başarılı", {
          model,
          ms: Math.round(performance.now() - t0),
          chars: synced.length,
        });
        return;
      }
      lastDetail = `[${model}] Boş yanıt — sonraki modele geçiliyor.`;
      debugLog("llm", "warn", "Boş yanıt", { model });
    } catch (err) {
      if (cb.signal?.aborted) throw err;
      if (isTimeoutError(err)) {
        lastDetail = `[${model}] Zaman aşımı (${PER_MODEL_TIMEOUT_MS / 1000}s)`;
        debugLog("llm", "warn", "Model zaman aşımı", {
          model,
          ms: Math.round(performance.now() - t0),
        });
        continue;
      }
      if (err instanceof OpenRouterError) {
        lastStatus = err.status;
        lastDetail = err.message;
        if (err.status === 429) saw429 = true;
        if (isRetryableStatus(err.status)) continue;
      }
      lastDetail = err instanceof Error ? err.message : "Bilinmeyen hata";
      debugLog("llm", "error", "Model hatası", { model, detail: lastDetail });
    } finally {
      cleanup();
    }
  }

  debugLog("llm", "error", "Tüm modeller başarısız", { lastStatus, lastDetail, saw429 });

  if (saw429) {
    throw new RateLimitError(
      lastDetail || "Tüm ücretsiz modeller rate limit/kota sınırında. Birkaç dakika bekleyin.",
    );
  }
  throw new OpenRouterError(lastStatus || 503, lastDetail || "Tüm ücretsiz modeller yanıt vermedi.");
}

/** Tedavi/tavsiye/yaşam tarzı önerisi isteyen sorular — LLM'e gitmeden yanıtlanır. */
export function isClinicalAdviceRequest(userMessage: string): boolean {
  const q = userMessage.toLowerCase();
  return /tedavi|ilaç|ilac|reçete|recete|doz|hangi ilac|tavsiye|öneri|oneri|ne yapmal|ne yapmali|yaşam tarz|yasam tarz|beslen|egzersiz|internetten|internete|araştır|arastir|google|webden/.test(
    q,
  );
}

export function looksLikeEmptyLlmRefusal(text: string): boolean {
  const t = text.trim().toLowerCase();
  if (t.length > 220) return false;
  return /yardımcı olamam|yardimci olamam|cannot help|can't help|unable to assist|sorry/i.test(t);
}

export function buildAdviceRefusalAnswer(ctx: AnalysisContext): string {
  const pct = (n: number) => `%${(n * 100).toFixed(1)}`;
  const primaryTr = PATHOLOGY_LABELS_TR[ctx.primary.label] || ctx.primary.label;
  const above = ctx.predictedLabels.filter((l) => l !== "NORM" && l !== ctx.primary.label);
  let locLine = "";
  if (ctx.localization) {
    const top = Object.entries(ctx.localization.probabilities).sort((a, b) => b[1] - a[1])[0];
    if (top) {
      locLine = `\n- **Lokalizasyon:** ${ctx.localization.labels_tr[top[0]] || top[0]} (${pct(top[1])})`;
    }
  }
  const guardLine = ctx.consistency
    ? `\n- **Model uyumu:** ${ctx.consistency.agreement} (triage: ${ctx.consistency.triage_level})`
    : "";

  return (
    `**Tedavi veya kişisel tavsiye veremem** — CardioGuard yalnızca bu EKG oturumundaki model çıktısını açıklar; internet araştırması veya ilaç/yaşam tarzı önerisi sunmaz.\n\n` +
    `**Bu oturumda ne görüyoruz?**\n` +
    `- **Birincil bulgu:** ${primaryTr} (${pct(ctx.primary.confidence)})\n` +
    (above.length ? `- **Eşik üstü diğer sınıflar:** ${above.join(", ")}\n` : "") +
    locLine +
    guardLine +
    `\n\n**Sonraki adım (genel):** Bu sonuçlar karar destek içindir; kesin değerlendirme, takip ve tedavi planı **hekim** tarafından yapılmalıdır.\n\n` +
    `İsterseniz lokalizasyon, güvenilirlik, STTC veya XAI kanıtları hakkında soru sorabilirsiniz.\n\n` +
    `_Bu bilgi karar destek amaçlıdır._`
  );
}

export function templateAnswer(userMessage: string, ctx: AnalysisContext): string {
  const q = userMessage.toLowerCase();
  const pct = (n: number) => `%${(n * 100).toFixed(1)}`;
  const tail = "\n\n_Bu bilgi karar destek amaçlıdır._";

  if (isClinicalAdviceRequest(userMessage)) {
    return buildAdviceRefusalAnswer(ctx);
  }

  if (/hasta dil|sade|basit/.test(q)) {
    return `**Hasta dilinde özet:** EKG'nizde kalp kasında dolaşım sorununa işaret eden bir bulgu (${
      PATHOLOGY_LABELS_TR[ctx.primary.label] || ctx.primary.label
    }) öne çıkıyor. Bilgisayar modeli bundan ${pct(
      ctx.primary.confidence,
    )} oranında emin. Bu kesin bir tanı değildir; doktorunuzun değerlendirmesi gerekir.${tail}`;
  }

  if (/lokaliz|bölge|bolge|asmi|nerede|hangi derivasyon/.test(q)) {
    if (!ctx.localization) return "Bu oturumda lokalizasyon verisi bulunmuyor." + tail;
    const top = Object.entries(ctx.localization.probabilities).sort((a, b) => b[1] - a[1])[0];
    const def = ctx.glossary[top[0]] || "";
    return `**Lokalizasyon (kaynak: localization):** En yüksek olasılık ${
      ctx.localization.labels_tr[top[0]] || top[0]
    } (${pct(top[1])}). ${def}${tail}`;
  }

  if (/güven|guven|ne kadar|risk|triyaj|triage/.test(q)) {
    const c = ctx.consistency;
    return `**Güvenilirlik:** Birincil bulgu güveni ${pct(ctx.primary.confidence)}.${
      c ? ` Consistency Guard: ${c.agreement}, triyaj seviyesi ${c.triage_level}.` : ""
    }${ctx.xai ? ` XAI coherence: ${pct(ctx.xai.coherence_score)}.` : ""}${tail}`;
  }

  if (/xai|gradcam|grad-cam|shap|açıkla.*göster|gosteriyor/.test(q)) {
    if (!ctx.xai) return "Bu oturumda XAI verisi bulunmuyor." + tail;
    return `**XAI (Grad-CAM):** ${ctx.xai.gradcam_summary}.\n\n**SHAP:** ${
      ctx.xai.shap_summary
    }.\n\nTutarlılık skoru: ${pct(ctx.xai.coherence_score)}.${tail}`;
  }

  if (/stt?c/.test(q)) {
    return `**STTC (kaynak: ensemble):** Olasılık ${pct(ctx.probabilities.STTC)} (eşik ${
      ctx.thresholds.STTC ? pct(ctx.thresholds.STTC) : "—"
    }), bu nedenle eşik üzerinde değerlendirildi. ${ctx.glossary.STTC}${tail}`;
  }

  for (const key of Object.keys(ctx.glossary)) {
    if (q.includes(key.toLowerCase())) {
      const prob = (ctx.probabilities as Record<string, number>)[key];
      return `**${key}:** ${ctx.glossary[key]}.${
        prob !== undefined ? ` Bu oturumdaki olasılık: ${pct(prob)}.` : ""
      }${tail}`;
    }
  }

  return `Bu oturumdaki analiz verisine göre birincil bulgu **${
    PATHOLOGY_LABELS_TR[ctx.primary.label] || ctx.primary.label
  }** (${pct(
    ctx.primary.confidence,
  )}). Daha spesifik bir soru sorabilirsiniz: lokalizasyon, güvenilirlik, STTC, XAI veya bir terim (MI, HYP...).${tail}`;
}

/** OpenRouter key + ücretsiz model zincirini test et. */
export async function testOpenRouterConnection(apiKey: string): Promise<{
  ok: boolean;
  model?: string;
  detail?: string;
}> {
  if (!apiKey.trim()) return { ok: false, detail: "API anahtarı boş" };

  let lastDetail = "";
  for (const model of freeModelChain()) {
    const { signal, cleanup } = createModelAbort(undefined);
    try {
      const res = await fetch(ENDPOINT, {
        method: "POST",
        signal,
        headers: openRouterHeaders(apiKey.trim()),
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: "Merhaba" }],
          max_tokens: 12,
          stream: false,
        }),
      });
      if (res.ok) {
        debugLog("llm", "info", "OpenRouter test OK", { model });
        return { ok: true, model };
      }
      lastDetail = `[${model}] ${await readErrorSnippet(res)} (HTTP ${res.status})`;
      if (isRetryableStatus(res.status)) continue;
    } catch (e) {
      lastDetail = e instanceof Error ? e.message : "Bağlantı hatası";
      if (isTimeoutError(e)) continue;
    } finally {
      cleanup();
    }
  }

  debugLog("llm", "error", "OpenRouter test failed", { detail: lastDetail });
  return { ok: false, detail: lastDetail || "Tüm ücretsiz modeller yanıt vermedi" };
}

export { DISCLAIMER };
