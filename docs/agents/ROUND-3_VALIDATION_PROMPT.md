# CardioGuard-AI — Tur 3 Doğrulama Prompt'u

**Oluşturulma:** 2026-06-14  
**Amaç:** Tur 1–2 implementasyonundan sonra hâlâ “açık” denilen 6 maddenin **gerçekten geçerli mi**, **ne kadar kritik**, **ne düzeyde çözülmüş** olduğunu kod kanıtıyla doğrulamak.  
**Ön koşul:** [`VALIDATION_REPORT.md`](./VALIDATION_REPORT.md) ve [`PROGRESS.md`](./work_packages/PROGRESS.md) okunmuş olmalı.

---

## Tur 3 — Doğrulanacak maddeler (ön snapshot)

Bu tablo **implementasyon sonrası hızlı tarama** — Plan modu agent **mutlaka kodla teyit edecek**.

| ID | Madde | Ön durum | Ön öncelik | Nerede bakılır |
|----|-------|----------|------------|----------------|
| **R3-01** | Dual superclass loader (`ECGCNN` vs `MultiLabelECGCNN`) | Muhtemelen **DOĞRULANDI** — API `load_model_safe`→`ECGCNN`, CLI `load_cnn_model`→`MultiLabelECGCNN` | P1 | `main.py` AppState, `model_loader.py`, `run_inference_superclass.py` |
| **R3-02** | Localization **eğitim raw** vs **inference z-score** | Muhtemelen **DOĞRULANDI** — `train_mi_localization.py` norm yok; inference standalone+embedded z-score | P0 klinik | `train_mi_localization.py`, `run_inference_localization.py`, `core_predict` |
| **R3-03** | `ci.yml` untracked → remote CI yok | Muhtemelen **DOĞRULANDI** — `git status` → `?? .github/workflows/ci.yml` | P0 süreç | `.github/workflows/ci.yml`, `git ls-files` |
| **R3-04** | XAI manifest 3 writer, şema farklı | Muhtemelen **DOĞRULANDI** | P2 | `xai/pipeline.py`, `xai/reporting.py`, `run_inference_localization.py` |
| **R3-05** | OpenRouter tarayıcıdan direkt (key exposure) | Muhtemelen **DOĞRULANDI** — tasarım tradeoff | P1 prod güvenlik | `frontend/src/lib/openrouter.ts`, `useAnalysisSession.ts` |
| **R3-06** | Sinyal I/O kopyalı; `signal_io.py` yok | Muhtemelen **DOĞRULANDI** — konsolidasyon yapılmadı | P2 teknik | `main.py` `parse_ecg_file`, `run_inference_*.py` `load_ecg_signal` |

### Tur 1–2’de kapatıldığı iddia edilenler (Tur 3’te **REDDEDİLDİ** olarak işaretle eğer hâlâ geçerli)

| ID | Ne yapıldı | Doğrulama yeri |
|----|------------|----------------|
| **R3-FIX-01** | Standalone localization + validate + z-score | `run_inference_localization.py` |
| **R3-FIX-02** | `validate_feature_schema` inference’da | `core_predict` + `main.py` feature_schema pass |
| **R3-FIX-03** | Zod API parse | `superclassSchema.ts`, `cardioguard.ts` |
| **R3-FIX-04** | Docker multi-stage + SPA routes | `Dockerfile`, `main.py` static/SPA |
| **R3-FIX-05** | Backend health UX | `index.tsx` WelcomeView |
| **R3-FIX-06** | MI-localization response parity (versions, latency, glossary) | `MILocalizationResponse`, `predict_mi_localization` |
| **R3-FIX-07** | Eval import paths | `run_comprehensive_test.py`, `test_evaluation_imports.py` |

---

## Kullanım

1. Cursor **Plan moduna** geç
2. Aşağıdaki `--- PROMPT BAŞLANGICI ---` … `--- PROMPT SONU ---` bloğunu **tamamen** yapıştır
3. Orchestrator 6 doğrulama agent spawn eder (+ 1 sentez)
4. Çıktı: `docs/agents/ROUND-3_VALIDATION_REPORT.md`
5. **Kod değiştirme, commit yapma**

---

## --- PROMPT BAŞLANGICI ---

Sen CardioGuard-AI repo'sunun **Tur 3 doğrulama orchestrator'ı**.

Görev: Tur 1–2 sonrası “hâlâ açık” denilen 6 maddeyi (R3-01 … R3-06) ve 7 fix teyidini (R3-FIX-01 … R3-FIX-07) **gerçek kod ve git durumu** üzerinde doğrula. Tahmin yasak — her iddia `path:line` veya `git`/`grep` çıktısı ile.

### Bağlam dosyaları

- `docs/agents/VALIDATION_REPORT.md` — Tur 1 doğrulama
- `docs/agents/work_packages/PROGRESS.md` — implementasyon log
- `docs/agents/ORCHESTRATION_PLAN.md` — mimari kurallar §1
- `CLAUDE.md` — aktif mimari

### Orchestrator kuralları

1. Alt agent'ları sırayla spawn et (V-R3-01 … V-R3-06 paralel OK; V-R3-FIX batch paralel OK; sonra V-R3-15 sentez)
2. Bulgu durumu: `DOĞRULANDI` | `REDDEDİLDİ` | `KISMEN` | `GÜNCELLENDİ`
3. **Şiddet skoru** (her R3-0X için zorunlu):

| Skor | Anlam |
|------|--------|
| **S0** | Kapatıldı / risk yok |
| **S1** | Düşük — teknik borç, prod etkisi minimal |
| **S2** | Orta — drift/UX/test riski |
| **S3** | Yüksek — yanlış tahmin, güvenlik, CI, klinik güven |
| **S4** | Kritik — aktif production bug veya veri güvenilirliği |

4. **Efor tahmini** fix için: `XS` (<2h) | `S` (1 gün) | `M` (2–3 gün) | `L` (1 hafta+) | `HITL` (insan kararı)
5. Yeni bulgular: `R3-NEW-XX`
6. **Commit yapma, kod değiştirme**

### Doğrulama komutları (mümkünse çalıştır, çıktıyı rapora yapıştır)

```bash
# Git / CI
git status .github/workflows/ci.yml
git ls-files .github/workflows/

# Backend tests
pytest tests/ -q --ignore=tests/test_data.py

# Frontend
cd frontend && npm test && npx tsc --noEmit

# Grep smoke
rg "load_model_safe|MultiLabelECGCNN|load_cnn_model" src/ --glob "*.py"
rg "_write_manifest|manifest.json" src/ --glob "*.py"
rg "parse_ecg_file|load_ecg_signal" src/ --glob "*.py"
rg "openrouter.ai" frontend/src/
test -f src/utils/signal_io.py && echo EXISTS || echo MISSING

# Localization train vs inference
rg "normalize|apply_superclass" src/pipeline/training/train_mi_localization.py
rg "apply_superclass_normalization" src/pipeline/inference/run_inference_localization.py src/pipeline/inference/run_inference_superclass.py
```

---

### Agent V-R3-01 — Dual loader

**Doğrula:** R3-01  
**Dosyalar:** `src/backend/main.py` (AppState), `src/utils/model_loader.py`, `src/pipeline/inference/run_inference_superclass.py`, `src/models/cnn.py`, `train_superclass_cnn.py`  
**Sorular:**

- API startup hangi sınıfı yüklüyor? CLI hangi sınıfı?
- Ağırlıklar gerçekten uyumlu mu (key remap yeterli mi)?
- `predict()` API path'te type mismatch riski var mı?
- Fix: tek loader mi, yoksa `MultiLabelECGCNN` her yerde mi?

**Çıktı:** Durum + S0–S4 + efor + kanıt satırları

---

### Agent V-R3-02 — Localization train vs inference norm

**Doğrula:** R3-02, R3-FIX-01 (kısmen mi?)  
**Dosyalar:** `train_mi_localization.py`, `run_inference_localization.py`, `run_inference_superclass.py` (`core_predict` embedded loc), `src/utils/signal.py`  
**Sorular:**

- Eğitim pipeline'da z-score var mı?
- Standalone endpoint normalize ediyor mu? (Tur 2 fix)
- Embedded path aynı tensor'ı kullanıyor mu?
- **Klinik risk:** model raw ile eğitildi, inference z-score — tahmin drift boyutu?
- Retrain gerekli mi yoksa inference'ı raw'a çekmek daha doğru mu?

**Çıktı:** Durum + S0–S4 + önerilen fix yönü (A: retrain z-score, B: inference raw, C: her ikisi align)

---

### Agent V-R3-03 — CI untracked

**Doğrula:** R3-03  
**Dosyalar:** `.github/workflows/ci.yml`, `git status`, `docs/agents/work_packages/PROGRESS.md` (WP-06 ✅ iddiası)  
**Sorular:**

- Dosya repoda tracked mı?
- Workflow içeriği yerel test komutlarıyla uyumlu mu?
- Remote'da CI çalışması için ne eksik (sadece commit/push)?
- PROGRESS “WP-06 done” ile çelişiyor mu?

**Çıktı:** Durum + S0–S4 + “commit yeterli mi?”

---

### Agent V-R3-04 — XAI manifest unify

**Doğrula:** R3-04  
**Dosyalar:** `src/xai/pipeline.py` `_write_manifest`, `src/xai/reporting.py`, `run_inference_localization.py` `_write_manifest`, `src/contracts/artifacts.py`  
**Sorular:**

- Kaç manifest writer? Şema farkları neler?
- API path `cards.jsonl` yazıyor mu? `extract_highlights` çalışıyor mu?
- Unify için minimum diff ne? Full unify vs adapter layer?
- Tur 2 `extract_sanity` RELIABLE fix yeterli mi?

**Çıktı:** Durum + S0–S4 + unify stratejisi önerisi

---

### Agent V-R3-05 — OpenRouter browser direct

**Doğrula:** R3-05  
**Dosyalar:** `frontend/src/lib/openrouter.ts`, `useAnalysisSession.ts`, `storage.ts`, `settings.tsx`  
**Sorular:**

- LLM çağrısı backend'den mi browser'dan mı?
- API key nerede saklanıyor? Prod risk seviyesi?
- Demo modu LLM'i kesiyor mu?
- Backend proxy fix eforu ve alternatifler (env-only key, rate limit)?

**Çıktı:** Durum + S0–S4 + prod vs demo ayrımı

---

### Agent V-R3-06 — Signal I/O duplication

**Doğrula:** R3-06  
**Dosyalar:** `main.py` `parse_ecg_file`, `run_inference_superclass.py` `load_ecg_signal`, `run_inference_binary.py` `load_ecg_signal`, `src/utils/signal.py`  
**Sorular:**

- `signal_io.py` var mı?
- Kaç kopya? Farklar (validate, npz keys, allow_pickle)?
- Konsolidasyon breaking change riski?
- Minimum fix: shared `load_upload_bytes` + `load_ecg_from_path`?

**Çıktı:** Durum + S0–S4 + önerilen modül sınırı

---

### Agent V-R3-FIX — Tur 2 fix teyit batch

**Doğrula:** R3-FIX-01 … R3-FIX-07 (her biri ayrı satır)  
**Yöntem:** Kod satırı + ilgili test dosyası çalışıyor mu  
**Çıktı:** Tablo: Fix ID | DOĞRULANDI/REDDEDİLDİ/KISMEN | kanıt | eksik kalan

---

### Agent V-R3-15 — Sentez

**Girdi:** V-R3-01 … V-R3-06 + V-R3-FIX raporları  
**Oluştur:** `docs/agents/ROUND-3_VALIDATION_REPORT.md`

**Rapor şablonu:**

```markdown
# CardioGuard-AI — Tur 3 Doğrulama Raporu
Tarih: [ISO]
Orchestrator: Tur 3 validation

## Executive summary (3–5 madde)

## Özet tablo
| ID | Durum | Şiddet S0–S4 | Efor | Kısa kanıt |

## Açık maddeler (R3-01 … R3-06) — detay

## Fix teyitleri (R3-FIX-01 … 07)

## Yeni bulgular (R3-NEW-XX)

## Öncelikli fix sırası (doğrulama sonrası)
1. ...
2. ...

## Tur 4 önerisi (implementasyon — bu oturumda YAPMA)
| Sıra | WP / konu | Gerekçe |

## Komut çıktıları özeti
```

**Bitiş:** Kullanıcıya özet — “X madde gerçekten açık (S3/S4), Y madde kapatıldı, Z yanlış alarm”

Başla: V-R3-01 … V-R3-06 + V-R3-FIX paralel spawn.

--- PROMPT SONU ---

---

## Kısa prompt (tek oturum)

```
CardioGuard Tur 3 doğrulama: docs/agents/ROUND-3_VALIDATION_PROMPT.md içindeki
R3-01…R3-06 ve R3-FIX-01…07 maddelerini readonly kod taramasıyla doğrula.
Her madde: DOĞRULANDI/REDDEDİLDİ/KISMEN + şiddet S0–S4 + path:line kanıtı + fix eforu.
docs/agents/ROUND-3_VALIDATION_REPORT.md oluştur. Kod değiştirme.
Öncelik: R3-02 localization train/inference, R3-01 dual loader, R3-03 ci.yml git.
```

---

## Dosya ilişkileri

```
VALIDATION_REPORT.md          ← Tur 1 (F-* bulgular)
PROGRESS.md                   ← Tur 1–2 implementasyon
ROUND-3_VALIDATION_PROMPT.md  ← Bu dosya (Tur 3 prompt)
        │
        ▼
ROUND-3_VALIDATION_REPORT.md  ← Plan modu sonrası oluşturulacak
        │
        ▼
Tur 4 implementasyon planı    ← Rapor onayı sonrası
```

---

## Beklenen sonuçlar (hipotez — doğrulama agent teyit edecek)

| Madde | Beklenen şiddet | Not |
|-------|-----------------|-----|
| R3-02 Localization train≠inference | **S3–S4** | En yüksek klinik risk; inference unify edildi ama eğitim raw |
| R3-01 Dual loader | **S2** | Çalışıyor olabilir ama drift/audit riski |
| R3-03 CI untracked | **S3** | Süreç — tek commit ile S0 |
| R3-04 XAI manifests | **S2** | UX/artifact parity, klinik tahmin etkisi düşük |
| R3-05 OpenRouter browser | **S2 prod / S0 demo** | Bilinçli tradeoff |
| R3-06 Signal I/O dup | **S2** | Bakım riski, aktif bug değil |

Bu hipotezler **kesin değil** — Plan modu raporu bunları kanıtla güncelleyecek.
