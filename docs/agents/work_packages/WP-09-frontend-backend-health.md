# WP-09: Frontend backend health UX (P1)

**Tip:** AFK | **Blocked by:** — 

## Problem

Welcome ekranı LLM/demo durumu gösteriyor ama **backend `/health` yok**. Kullanıcı dosya yükleyip analiz fail olunca öğreniyor.

## Okunacak dosyalar

1. `frontend/src/routes/index.tsx` — WelcomeView, Sistem Durumu
2. `frontend/src/lib/api/cardioguard.ts`
3. `frontend/src/lib/storage.ts` — `getBackendUrl`
4. `docs/qa_manual_tr.md`

## Yapılacaklar

1. `cardioguard.ts`'ye `fetchHealth(baseUrl)` → GET `/health`
2. Welcome mount'ta poll (veya tek sefer):
   - Yeşil: "Backend bağlı"
   - Kırmızı: "Backend erişilemiyor — Ayarlar'dan URL kontrol edin"
3. Analyze butonu backend down iken disabled + tooltip
4. Opsiyonel: Settings'te "Backend bağlantısını test et" butonu

## Kabul kriterleri

- [ ] Welcome backend durumunu gösterir
- [ ] Backend kapalıyken upload/analyze engellenir veya net uyarı
- [ ] tsc + vitest geçer

```bash
cd frontend && npm test && npx tsc --noEmit
```
