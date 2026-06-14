# WP-13: LLM timeout ve UX düzeltmeleri (P1)

**Tip:** AFK | **Blocked by:** — 

## Problem

- Model başına 22s × 5 model = ~110s worst case
- `LlmStatusBanner` eski "openrouter/free ayarları" metni — model picker yok
- Settings test butonu env key varken disabled olabilir

## Okunacak dosyalar

1. `frontend/src/lib/openrouter.ts`
2. `frontend/src/hooks/useAnalysisSession.ts`
3. `frontend/src/components/chat/LlmStatusBanner.tsx`
4. `frontend/src/routes/settings.tsx`

## Yapılacaklar

1. Global LLM budget: `TOTAL_LLM_TIMEOUT_MS = 45_000` — aşılınca kural tabanlıya geç
2. Banner metnini güncelle: "Ücretsiz model zinciri otomatik"
3. Settings test: `getApiKey()` env fallback ile çalışsın
4. vitest: `isClinicalAdviceRequest`, `looksLikeEmptyLlmRefusal`

## Kabul kriterleri

- [ ] Max ~45s sonra yanıt (kural tabanlı fallback)
- [ ] Banner/settings metinleri doğru
- [ ] En az 3 yeni vitest
