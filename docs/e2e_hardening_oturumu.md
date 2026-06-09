# CardioGuard E2E Sertleştirme Oturumu Özeti

**Tarih:** 2026-06-09  
**Durum:** Tamamlandı, commit için hazır

## Ne yapıyorduk?

CardioGuard uçtan uca akışında (EKG yükleme → ensemble tahmin → XAI → klinik sohbet) güvenilirlik, veri tutarlılığı ve kullanıcı deneyimi sorunlarını faz faz düzeltiyorduk. Son oturumda iki ek konu vardı:

1. **Tarayıcı testlerinin görünürlüğü** — Agent'ın kullanıcının yerel testlerini okuyabilmesi
2. **LLM tavsiye sorusu** — "tavsiye/öneri/internetten araştır" sorularında modelin kuru "yardımcı olamam" demesi

---

## Tamamlanan fazlar (0–6)

| Faz | Konu | Özet |
|-----|------|------|
| 0 | Teşhis | Backend inference, XAI PNG, latency doğrulandı |
| 1 | OpenRouter / sohbet | Kaynak rozeti (LLM / Otomatik / Kural), hata toast'ları, ücretsiz model zinciri, test butonu |
| 2 | Hız güveni | `latency_ms` API'den, min ~2s progress, Canlı vs Simülasyon rozeti |
| 3 | Veri tutarlılığı | Ensemble default 0.15 CNN / 0.85 XGB, divergence uyarısı, session schema v2 |
| 4 | XAI | matplotlib Agg fix (boş Grad-CAM), gerçek PNG serve, SHAP etiketleri, coherence kalibrasyonu |
| 5 | Consistency Guard | Okunabilir AGREE_MI açıklaması + karar tablosu |
| 6 | Testler | pytest + vitest, `docs/qa_manual_tr.md` manuel kontrol listesi |

---

## Son oturumda eklenenler

### 1. Oturum debug logu

| Bileşen | Dosya / endpoint |
|---------|------------------|
| Frontend logger | `frontend/src/lib/sessionDebugLog.ts` |
| Backend yazma | `POST /debug/client-log` |
| Backend okuma | `GET /debug/client-log?tail=N` |
| Log dosyası | `logs/client-events.jsonl` (gitignore'da) |

Agent veya geliştirici şu komutla son olayları okuyabilir:

```bash
curl "http://localhost:8000/debug/client-log?tail=30"
```

Loglanan olaylar: analiz başlangıç/bitiş, kullanıcı mesajı, LLM model denemeleri, süre, hatalar. **API anahtarı asla loglanmaz.**

### 2. LLM takılma önleme

- Model başına **22 saniye** zaman aşımı → sonraki ücretsiz modele geçiş
- Chat'te hangi model denendiği metni (`llmProgress`)
- `openrouter.ts`: `FREE_MODEL_CHAIN` (5 ücretsiz model)

### 3. Tavsiye / tedavi sorusu düzeltmesi

**Sorun:** Kullanıcı *"bu EKG'ye sahip kişiye tavsiyeler neler, internetten araştır"* dediğinde LLM badge ile gelen cevap: *"Üzgünüm, bu konuda size yardımcı olamam."*

**Kök neden:** Soru doğrudan LLM'e gidiyordu; model güvenlik filtresi devreye girdi. Kural tabanlı katman sadece `tedavi|ilaç` gibi dar regex'e bakıyordu; `tavsiye|öneri|internetten` yakalanmıyordu.

**Düzeltme:**

- `isClinicalAdviceRequest()` — genişletilmiş anahtar kelime listesi
- `buildAdviceRefusalAnswer()` — tedavi veremeyeceğini söyler **ama** oturum bulgularını (birincil sınıf, lokalizasyon, guard) özetler
- LLM'e gitmeden önce kural tabanlı yanıt (`source: rule`)
- LLM yine de kuru red dönerse `looksLikeEmptyLlmRefusal()` ile kural tabanlıya çevirme
- System prompt güncellendi: boş red yerine bulgu özeti isteniyor

---

## Loglardan görülen son test (2026-06-09)

Backend yeniden başlatıldıktan sonra `logs/client-events.jsonl` içinde:

1. `test_mi_sample.npz` — canlı analiz ~1s, primary **MI**
2. "LLM ile detaylandır" — `openrouter/free`, ~5s, 952 karakter — **başarılı**

Tavsiye sorusu bu log dosyasında henüz yok (muhtemelen debug endpoint aktif olmadan önce sorulmuş).

---

## Çalıştırma

```bash
# Backend
pip install -r requirements.txt
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend
cd frontend && npm install && npm run dev
```

## Test

```bash
.venv/bin/python -m pytest tests/ -q
cd frontend && npm test && npx tsc --noEmit
```

---

## Bilinen sınırlar

- OpenRouter ücretsiz kota (~200/gün) gerçek; limit dolunca kural tabanlıya düşer
- Agent tarayıcı konsolunu doğrudan okuyamaz; `GET /debug/client-log` kullanılmalı
- `datetime.utcnow()` / `torch.load weights_only` / lifespan migration — CLAUDE.md'deki teknik borç listesinde

---

## Değişen ana dosyalar (son fix dahil)

```
frontend/src/lib/sessionDebugLog.ts      # yeni
frontend/src/lib/openrouter.ts           # timeout, tavsiye filtresi, debug
frontend/src/hooks/useAnalysisSession.ts # llmProgress, pre-check
frontend/src/components/chat/ClinicalChatPanel.tsx
frontend/src/components/chat/ChatMessage.tsx  # rule badge
frontend/src/lib/types.ts                # MessageSource + rule
src/backend/main.py                      # /debug/client-log
tests/test_api.py                        # debug log test
.gitignore                               # client-events.jsonl, xai runs
```
