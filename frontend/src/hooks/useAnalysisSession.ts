import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import type {
  AnalysisContext,
  AnalyzeOptions,
  AppState,
  ChatMessage,
  LlmStatus,
  MessageSource,
  StoredSession,
} from "@/lib/types";
import { predictSuperclass } from "@/lib/api/cardioguard";
import { createMockContext, buildAutoSummary } from "@/lib/mockData";
import { mapResultToContext } from "@/lib/mapResultToContext";
import {
  clearSession,
  getApiKey,
  getBackendUrl,
  getDemoMode,
  loadSession,
  saveSession,
} from "@/lib/storage";
import {
  OpenRouterError,
  RateLimitError,
  isClinicalAdviceRequest,
  looksLikeEmptyLlmRefusal,
  streamChat,
  templateAnswer,
} from "@/lib/openrouter";
import { debugLog } from "@/lib/sessionDebugLog";

export const ANALYSIS_STEPS = [
  "EKG sinyali okunuyor...",
  "CNN + XGBoost ensemble çalışıyor...",
  "Consistency Guard kontrol ediliyor...",
  "MI lokalizasyonu hesaplanıyor...",
  "XAI açıklamaları üretiliyor...",
  "Otomatik klinik özet hazırlanıyor...",
];

const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);

function openRouterErrorMessage(status: number): string {
  switch (status) {
    case 401:
      return "OpenRouter API anahtarı geçersiz. Ayarlar'dan kontrol edin.";
    case 402:
      return "OpenRouter hesabında kredi gerekiyor veya ücretsiz kota bitti.";
    case 403:
      return "OpenRouter erişimi reddedildi (model/anahtar izni yok).";
    case 404:
      return "Seçili model bulunamadı. Ayarlar'dan modeli değiştirin.";
    default:
      return `OpenRouter hatası (HTTP ${status}). Kural tabanlı yanıta geçildi.`;
  }
}

function runMockPipeline(
  fileName: string,
  onStep: (index: number) => void,
  onDone: (ctx: AnalysisContext) => void,
) {
  const ctx = createMockContext(fileName);
  let i = 0;
  const interval = setInterval(() => {
    i += 1;
    if (i >= ANALYSIS_STEPS.length) {
      clearInterval(interval);
      setTimeout(() => onDone(ctx), 500);
    } else {
      onStep(i);
    }
  }, 650);
  return () => clearInterval(interval);
}

export function useAnalysisSession() {
  const [appState, setAppState] = useState<AppState>("welcome");
  const [context, setContext] = useState<AnalysisContext | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isDemo, setIsDemo] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [restored, setRestored] = useState(false);
  const [llmStatus, setLlmStatus] = useState<LlmStatus>("offline");
  const [isResponding, setIsResponding] = useState(false);
  const [llmProgress, setLlmProgress] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const stored = loadSession();
    if (stored) {
      setContext(stored.context);
      setMessages(stored.messages);
      setIsDemo(stored.isDemo);
      setAppState("results");
      setRestored(true);
    }
    setLlmStatus(getApiKey() ? "active" : "offline");
  }, []);

  const persist = useCallback(
    (ctx: AnalysisContext, msgs: ChatMessage[], demo: boolean) => {
      const s: StoredSession = {
        context: ctx,
        messages: msgs,
        isDemo: demo,
        timestamp: Date.now(),
      };
      saveSession(s);
    },
    [],
  );

  const finishAnalysis = useCallback(
    (ctx: AnalysisContext, demo: boolean) => {
      const summary: ChatMessage = {
        id: uid(),
        role: "assistant",
        content: buildAutoSummary(ctx),
        source: "auto",
      };
      setContext(ctx);
      setMessages([summary]);
      setIsDemo(demo);
      setAppState("results");
      setLlmStatus(getApiKey() && !demo && !getDemoMode() ? "active" : "offline");
      persist(ctx, [summary], demo);
    },
    [persist],
  );

  const runAnalysis = useCallback(
    (file: File | null, fileName: string, opts: AnalyzeOptions, demo: boolean) => {
      setRestored(false);
      setAppState("analyzing");
      setStepIndex(0);

      if (demo || !file) {
        debugLog("analysis", "info", "Demo/simülasyon analizi başladı", { fileName });
        runMockPipeline(fileName, setStepIndex, (ctx) => finishAnalysis(ctx, true));
        return;
      }

      debugLog("analysis", "info", "Canlı analiz başladı", { fileName, opts });

      const startedAt = Date.now();
      const MIN_DISPLAY_MS = 2000;
      let i = 0;
      const stepTimer = setInterval(() => {
        i += 1;
        if (i < ANALYSIS_STEPS.length) setStepIndex(i);
      }, 650);

      predictSuperclass(file, opts, getBackendUrl())
        .then((apiResult) => {
          clearInterval(stepTimer);
          setStepIndex(ANALYSIS_STEPS.length - 1);
          const ctx = mapResultToContext(apiResult, fileName, getBackendUrl());
          debugLog("analysis", "info", "Canlı analiz tamamlandı", {
            fileName,
            primary: ctx.primary.label,
            latencyMs: apiResult.latency_ms,
          });
          // Keep the progress visible at least MIN_DISPLAY_MS so a sub-second
          // backend response does not feel like a fake/mock run.
          const remaining = Math.max(400, MIN_DISPLAY_MS - (Date.now() - startedAt));
          setTimeout(() => finishAnalysis(ctx, false), remaining);
        })
        .catch((err: unknown) => {
          clearInterval(stepTimer);
          setAppState("welcome");
          const msg = err instanceof Error ? err.message : "Bilinmeyen hata";
          debugLog("analysis", "error", "Analiz başarısız", { fileName, error: msg });
          toast.error(`Analiz başarısız: ${msg}`);
        });
    },
    [finishAnalysis],
  );

  const loadDemo = useCallback(
    () =>
      runAnalysis(
        null,
        "ornek_ekg_record_00021.npy",
        { explain: true, sanityCheck: true, ensemble: 0.85 },
        true,
      ),
    [runAnalysis],
  );

  const reset = useCallback(() => {
    abortRef.current?.abort();
    clearSession();
    setContext(null);
    setMessages([]);
    setAppState("welcome");
    setRestored(false);
    setIsResponding(false);
    setLlmProgress(null);
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || !context || isResponding) return;

      const userMsg: ChatMessage = { id: uid(), role: "user", content: trimmed };
      const assistantId = uid();
      const placeholder: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        pending: true,
      };

      const history = messages;
      const baseMsgs = [...messages, userMsg, placeholder];
      setMessages(baseMsgs);
      setIsResponding(true);
      debugLog("ui", "info", "Kullanıcı mesajı", { preview: trimmed.slice(0, 120) });

      const apiKey = getApiKey();
      const demoForced = getDemoMode();

      const finalize = (content: string, source: MessageSource, status?: LlmStatus) => {
        setLlmProgress(null);
        setMessages((prev) => {
          const next = prev.map((m) =>
            m.id === assistantId ? { ...m, content, pending: false, source } : m,
          );
          persist(context, next, isDemo);
          return next;
        });
        if (status) setLlmStatus(status);
        setIsResponding(false);
      };

      if (!apiKey || demoForced || isDemo) {
        const answer = templateAnswer(trimmed, context);
        await new Promise((r) => setTimeout(r, 450));
        finalize(answer, "template", "offline");
        return;
      }

      if (isClinicalAdviceRequest(trimmed)) {
        debugLog("ui", "info", "Tavsiye sorusu — kural tabanlı yanıt", { preview: trimmed.slice(0, 80) });
        const answer = templateAnswer(trimmed, context);
        await new Promise((r) => setTimeout(r, 350));
        finalize(answer, "rule", "active");
        return;
      }

      const controller = new AbortController();
      abortRef.current = controller;
      let acc = "";
      try {
        setLlmStatus("active");
        await streamChat(apiKey, trimmed, context, history, {
          signal: controller.signal,
          onProgress: setLlmProgress,
          onToken: (tok) => {
            acc += tok;
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, content: acc } : m)),
            );
          },
        });
        if (acc) {
          if (looksLikeEmptyLlmRefusal(acc)) {
            debugLog("llm", "warn", "LLM boş red — kural tabanlıya çevrildi");
            finalize(templateAnswer(trimmed, context), "rule", "active");
          } else {
            finalize(acc, "llm", "active");
          }
        } else finalize(templateAnswer(trimmed, context), "template", "offline");
      } catch (err) {
        if (controller.signal.aborted) {
          setIsResponding(false);
          setLlmProgress(null);
          return;
        }
        if (err instanceof RateLimitError) {
          const detail = err.message || "OpenRouter ücretsiz kota/rate limit aşıldı.";
          toast.error(
            `${detail} Ayarlar'da modeli openrouter/free yapın veya birkaç dakika bekleyin.`,
          );
          finalize(templateAnswer(trimmed, context), "template", "limit");
        } else if (err instanceof OpenRouterError) {
          toast.error(openRouterErrorMessage(err.status));
          finalize(templateAnswer(trimmed, context), "template", "error");
        } else {
          const msg = err instanceof Error ? err.message : "Bilinmeyen hata";
          toast.error(`LLM yanıtı alınamadı: ${msg}. Kural tabanlı yanıta geçildi.`);
          finalize(templateAnswer(trimmed, context), "template", "error");
        }
      }
    },
    [context, messages, isResponding, isDemo, persist],
  );

  const elaborateWithLlm = useCallback(async () => {
    if (!context || isResponding) return;
    const apiKey = getApiKey();
    if (!apiKey || getDemoMode() || isDemo) {
      toast.error("LLM kullanılamıyor: API anahtarı yok veya demo modu açık.");
      return;
    }

    const assistantId = uid();
    const placeholder: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      pending: true,
      source: "llm",
    };
    const history = messages;
    setMessages((prev) => [...prev, placeholder]);
    setIsResponding(true);
    debugLog("ui", "info", "LLM ile detaylandır tıklandı");

    const prompt =
      "Bu EKG analizini bir hekime sunar gibi detaylı, yapılandırılmış biçimde yorumla: " +
      "birincil bulgu, eşik üstü diğer sınıflar, lokalizasyon, model uyumu (Consistency Guard) ve " +
      "XAI (Grad-CAM/SHAP) kanıtlarını madde madde açıkla; sonunda klinik bağlamda ne anlama geldiğini özetle.";

    const finalize = (content: string, source: MessageSource, status: LlmStatus) => {
      setLlmProgress(null);
      setMessages((prev) => {
        const next = prev.map((m) =>
          m.id === assistantId ? { ...m, content, pending: false, source } : m,
        );
        persist(context, next, isDemo);
        return next;
      });
      setLlmStatus(status);
      setIsResponding(false);
    };

    const controller = new AbortController();
    abortRef.current = controller;
    let acc = "";
    try {
      setLlmStatus("active");
      await streamChat(apiKey, prompt, context, history, {
        signal: controller.signal,
        onProgress: setLlmProgress,
        onToken: (tok) => {
          acc += tok;
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: acc } : m)),
          );
        },
      });
      if (acc) finalize(acc, "llm", "active");
      else finalize(templateAnswer(prompt, context), "template", "offline");
    } catch (err) {
      if (controller.signal.aborted) {
        setIsResponding(false);
        setLlmProgress(null);
        return;
      }
      if (err instanceof RateLimitError) {
        toast.error("OpenRouter ücretsiz limiti doldu. Kural tabanlı yanıta geçildi.");
        finalize(templateAnswer(prompt, context), "template", "limit");
      } else if (err instanceof OpenRouterError) {
        toast.error(openRouterErrorMessage(err.status));
        finalize(templateAnswer(prompt, context), "template", "error");
      } else {
        const msg = err instanceof Error ? err.message : "Bilinmeyen hata";
        toast.error(`LLM yanıtı alınamadı: ${msg}.`);
        finalize(templateAnswer(prompt, context), "template", "error");
      }
    }
  }, [context, messages, isResponding, isDemo, persist]);

  const llmAvailable = !!getApiKey() && !getDemoMode() && !isDemo;

  return {
    appState,
    context,
    messages,
    isDemo,
    stepIndex,
    restored,
    llmStatus,
    isResponding,
    llmProgress,
    llmAvailable,
    runAnalysis,
    loadDemo,
    reset,
    sendMessage,
    elaborateWithLlm,
    dismissRestored: () => setRestored(false),
  };
}
