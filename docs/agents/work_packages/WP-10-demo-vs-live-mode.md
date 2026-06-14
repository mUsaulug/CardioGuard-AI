# WP-10: Demo mod vs canlı analiz ayrımı (P1)

**Tip:** AFK | **Blocked by:** — 

## Problem

Settings **Demo Modu** (`getDemoMode()`) açıkken gerçek dosya yüklense bile `runMockPipeline` çalışıyor — kullanıcı canlı sandığı halde simülasyon.

## Okunacak dosyalar

1. `frontend/src/hooks/useAnalysisSession.ts` — `runAnalysis`, `sendMessage`
2. `frontend/src/routes/index.tsx`, `settings.tsx`
3. `frontend/src/lib/storage.ts`

## Yapılacaklar

1. Demo mod sadece **LLM'i** kapatmalı veya ayrı "Simülasyon analizi" butonu olmalı
2. Gerçek dosya + demo mod açık → toast uyarı: "Demo modu: simülasyon kullanılıyor" veya demo'yu otomatik kapat
3. Evidence panel'de badge net: **Canlı analiz** vs **Simülasyon**
4. `loadDemo()` ayrı kalır — tek tık demo

## Kabul kriterleri

- [ ] Gerçek upload + demo off → backend predict
- [ ] Demo açık + upload → kullanıcı ne olduğunu anlar (mock veya uyarı)
- [ ] vitest geçer
