# CardioGuard-AI — Uygulama Planı

Tek sayfalık klinik dashboard + akıllı chat asistanı. React + TypeScript + Tailwind (TanStack Start). Varsayılan dil **Türkçe**. Backend gerekmez — tümü frontend, OpenRouter tarayıcıdan, mock veri ile demo.

## Genel Akış (4 durum)
- **WELCOME** → ilk yüklemede karşılama ekranı (otomatik demo YOK). Upload zone + "Demo Analizi Gör" CTA + ayarlar.
- **ANALYZING** → ~4 sn simülasyonlu 6 adımlı pipeline animasyonu + skeleton.
- **RESULTS** → 2 kolon: sol "Kanıt Paneli", sağ "Klinik Asistan".
- **FALLBACK modları** → LLM Aktif / Limit Aşıldı (429) / Demo (key yok) / Backend Offline.

## Tasarım Sistemi
- Klinik + modern AI estetiği (Notion/Linear cilası). Medical blue `#2563EB`, success `#10B981`, warning `#F59E0B`, danger `#EF4444`. Açık/koyu tema.
- Tipografi: Inter/DM Sans başlık, Inter gövde, JetBrains Mono teknik. Rounded-xl kartlar, yumuşak gölge.
- Renkler hep metin etiketiyle birlikte (erişilebilirlik). Lucide ikonlar. Framer Motion ile stagger fade-in, typing indicator, progress animasyonu, primary kart pulse.
- Tüm token'lar `src/styles.css` içinde tanımlanır; component'lerde semantik sınıflar kullanılır.

## Rotalar
- `/` — ana analiz + chat (tek görünüm)
- `/settings` — OpenRouter API key (localStorage), demo modu, tema, backend URL
- `/about` — tez projesi bilgisi, metodoloji, disclaimer

## Sol Kolon — Kanıt Paneli (~%45, sticky)
- **A) Oturum başlığı**: dosya adı, zaman, kısaltılmış model hash
- **EcgWaveformMock**: stilize 12-derivasyon SVG dalga formu (I, II, III, aVR, aVL, aVF, V1–V6)
- **B) Birincil Tanı kartı**: büyük etiket + güven rozeti (renk kodlu >80 yeşil / 50–80 amber / <50 kırmızı), çoklu-etiket çipleri, "Kural: MI-first-then-priority"
- **C) Olasılık Chart**: MI/STTC/CD/HYP/NORM yatay bar, eşik çizgileri, CNN/XGB/Ensemble görünüm toggle
- **D) Consistency Guard kartı**: agreement türü, triyaj rozeti (YÜKSEK/ORTA/DÜŞÜK/İNCELEME), Superclass vs Binary MI yan yana, uyarılar
- **E) MI Lokalizasyon kartı**: 5 bölge (AMI/ASMI/ALMI/IMI/LMI) Türkçe etiket + olasılık + basit kalp diyagramı vurgusu
- **F) XAI accordion**: Grad-CAM placeholder, Unified narrative (markdown), SHAP özet, sanity rozeti, coherence gauge (0–100)
- **G) Teknik Detaylar** (varsayılan kapalı): pretty-print JSON viewer + kopyala butonu

## Sağ Kolon — Klinik Asistan (~%55, full-height) — DEMONUN YILDIZI
- **Header**: medical AI avatar, başlık, alt başlık, durum pill ("LLM Aktif" / "Offline Mod" / "Limit Aşıldı")
- **Otomatik ilk mesaj**: analiz bitince asistan otomatik klinik özet gönderir (📋 Klinik Özet + disclaimer)
- **Hızlı yanıt çipleri**: "MI ne anlama geliyor?", "STTC neden tespit edildi?", "Lokalizasyonu açıkla", "Hasta dilinde özetle", "Risk seviyesi nedir?", "XAI ne gösteriyor?"
- **Chat input**: placeholder, Enter ile gönder, karakter limiti göstergesi
- **ChatMessage**: markdown render, streaming typing efekti, assistant=sol gri-mavi / user=sağ primary
- **Hekim dili / Hasta dili** modları kullanıcı isteğiyle

## Asistan Kuralları (sistem prompt'a gömülü)
- Her istekte tam `AnalysisContext` JSON gönderilir; asistan SADECE bu veriyi açıklar.
- Olasılıkları değiştirmez, yeni patoloji eklemez, tanı koymaz ("karar destek").
- Tedavi/ilaç/tanı istenirse reddeder: "Bu sistem tanı koymaz ve tedavi önermez; yalnızca analiz sonuçlarını açıklar."
- İnternet/arama yok. Türkçe glossary (light RAG) sistem prompt'ta gömülü.
- Veri kaynağı belirtir ("Ensemble modeline göre...", "Grad-CAM analizine göre...").
- Consistency Guard = REVIEW ise belirsizliği vurgular.

## OpenRouter Entegrasyonu
- Tarayıcıdan direkt `https://openrouter.ai/api/v1/chat/completions`
- Model: `qwen/qwen3-next-80b-a3b-instruct:free`, fallback `google/gemma-4-31b-it:free`
- Headers: Authorization Bearer (Settings/localStorage'dan), HTTP-Referer, X-Title: CardioGuard-AI
- Streaming (token token), `max_tokens: 800`, `temperature: 0.3`
- API key: Settings'te girilir, localStorage'da saklanır (tez demosu için kabul edildi). `VITE_OPENROUTER_API_KEY` de desteklenir.
- **429/limit** → sarı banner + kural-tabanlı template fallback (glossary + context).
- **Key yok** → keyword-eşleşmeli akıllı görünümlü canned yanıtlar.

## Oturum Kalıcılığı (localStorage)
- Son oturum saklanır: analysis context, chat history, demo flag, timestamp.
- Reload'da <24s ise otomatik geri yükle + banner "Önceki oturum geri yüklendi".
- "Yeni Analiz" → storage temizler, WELCOME'a döner.
- İlk ziyaret (saklı oturum yok) → WELCOME, otomatik demo yok.

## Mock Veri
Spec'teki MI örneği (primary MI %87.3, STTC, ASMI lokalizasyon, AGREE_MI, coherence %82) gömülü; "Demo Analizi Gör" anında yükler, "Simülasyon — örnek veri" etiketi gösterir.

## Responsive
- ≥1024px: 2 kolon, chat sticky full-height
- Tablet: dikey stack (kanıt üstte, chat altta)
- Mobil: sekme switcher "Sonuçlar | Asistan"

## Erişilebilirlik
- Renk rozetlerinde metin etiketi, klavye ile gezinilebilir chat, ARIA etiketli grafikler, yüksek kontrast uyumu.

## Bileşen Listesi
AppShell, UploadZone, AnalysisProgress, EvidencePanel, EcgWaveformMock, PrimaryDiagnosisCard, ProbabilityChart, ConsistencyGuardCard, LocalizationCard, XaiAccordion, ClinicalChatPanel, ChatMessage, QuickReplyChips, LlmStatusBanner, DisclaimerFooter, DemoModeBanner
Hooklar: `useAnalysisSession` (context + history + localStorage), `useOpenRouterChat` (LLM + fallback)

---

### Teknik Notlar
- TanStack Start file-based routing: `src/routes/index.tsx`, `settings.tsx`, `about.tsx`. State machine (welcome/analyzing/results) `/` içinde yönetilir.
- `AnalysisContext` ve `ChatMessage` tipleri `src/lib/types.ts`; glossary + mock veri `src/lib/`.
- OpenRouter çağrısı tamamen client-side (`src/hooks/useOpenRouterChat.ts`), streaming için fetch ReadableStream parse.
- Markdown render için hafif bir kütüphane (react-markdown) eklenecek.
- Backend/Supabase/edge function YOK — saf frontend prototip.