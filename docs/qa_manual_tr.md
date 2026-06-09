# CardioGuard-AI — Manuel QA Kontrol Listesi (TR)

Uçtan uca sertleştirme sonrası kabul doğrulaması. Sıra önemli.

## Hazırlık
1. Backend: `.venv/bin/uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload`
2. Frontend: `cd frontend && npm run dev`
3. Tarayıcıda `localStorage`'ı temizle (eski oturum/şema karışmasın).

## A. LLM gerçekten çalışıyor mu (en kritik)
- [ ] Ayarlar → OpenRouter API anahtarını gir → **Kaydet** (kaydetmeden çıkarsan "Kaydedilmemiş değişiklikler" uyarısı görünmeli).
- [ ] Demo Modu **KAPALI**.
- [ ] Bir EKG yükle, Analiz Et. İlk mesajda **"Otomatik özet"** rozeti olmalı (LLM değil).
- [ ] Sohbete soru yaz → cevap balonunda **"LLM"** rozeti + Network sekmesinde `openrouter.ai/api/v1/chat/completions` isteği görünmeli.
- [ ] Hazır soru çiplerinden birine tıkla → akışlı (stream) detaylı cevap gelmeli.
- [ ] Anahtarı bozuk gir → sohbet et → **toast** ile "API anahtarı geçersiz" + balonda "Kural tabanlı" rozeti.
- [ ] İlk özet altındaki **"LLM ile detaylandır"** → detaylı LLM yorumu stream etmeli.

## B. XAI artifact (Grad-CAM/SHAP boş değil)
- [ ] Analizde "XAI Açıklama (explain=true)" işaretli olsun.
- [ ] Kanıt Paneli → XAI → **Grad-CAM** sekmesinde gerçek **PNG** görünmeli (CSS placeholder değil).
- [ ] "Tam boyutta aç" linki PNG'yi yeni sekmede açmalı.
- [ ] SHAP özetinde **"CNN gömme boyutu N"** gibi okunabilir etiketler olmalı (`feature_13` değil).
- [ ] Coherence göstergesi **%100 olmamalı** (kalibre edilmiş, ~%50-95 aralığı).

## C. Veri tutarlılığı
- [ ] Olasılık Dağılımı varsayılan **ensemble**; birincil kart % = ensemble MI %.
- [ ] **CNN** sekmesine geç → birincil karttan farklıysa sarı uyarı kutusu çıkmalı.
- [ ] Welcome'da model ağırlığı sliderı "XGB %85 · CNN %15" göstermeli.

## D. Hız / canlılık
- [ ] Canlı analizde ilerleme en az ~2 sn görünür kalmalı (mock hissi yok).
- [ ] Kanıt Paneli üst rozet: **"Canlı analiz · backend X.XXs"** (demo'da "Simülasyon").

## E. Consistency Guard
- [ ] `AGREE_MI` kodunun yanında insanca açıklama ("İki model de MI POZİTİF...").
- [ ] Karar tablosu: Superclass / Binary satırları + "Sonuç: Kararlar uyumlu/farklı".

## F. Oturum
- [ ] Sayfayı yenile → 24 saat içindeyse oturum geri yüklenir ("Önceki oturum geri yüklendi").
- [ ] Eski şemalı (v1) oturum varsa sessizce atlanır, çökme olmaz.

## Otomatik testler
```bash
# Backend
.venv/bin/python -m pytest            # 129 passed, 1 skipped (PTB-XL yoksa)
# Frontend
cd frontend && npm test               # vitest: mapResultToContext
cd frontend && npx tsc --noEmit       # tip kontrolü
```
