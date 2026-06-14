# CardioGuard-AI — Agent Orkestrasyon Planı

**Oluşturulma:** 2026-06-14  
**Amaç:** Repo'yu domain bazında ayrı agent oturumlarıyla taramak, bulguları doğrulamak ve WP backlog ile eşleştirmek.  
**Doğrulama:** Bu plan tamamlandıktan sonra `VALIDATION_PROMPT.md` içindeki prompt ile Plan modunda ikinci tur doğrulama çalıştırılır.

---

## 1. Proje Özeti (Faz 0 — tamamlandı)

CardioGuard-AI: 12 derivasyonlu EKG'den kardiyak patoloji tespiti (PTB-XL). Production stack:

| Katman | Teknoloji | Aktif giriş |
|--------|-----------|-------------|
| Frontend | TanStack Start, React 19 | `frontend/src/routes/index.tsx` |
| API | FastAPI (`src/backend/main.py`) | `POST /predict/superclass` |
| Inference | `run_inference_superclass.py` | `core_predict()` |
| XAI | `src/xai/pipeline.explain()` | `explain=true` query param |
| Modeller | CNN + XGB OVR ensemble | `artifacts/thresholds_superclass.json` |

### Mimari kurallar (tüm agentlar için zorunlu)

1. `main.py` içinde ML kodu **yok** — sadece `pipeline_predict()` çağrısı
2. Tek inference kaynağı: `src/pipeline/inference/run_inference_superclass.py`
3. XAI artifact'ları pipeline üretir; backend manifest okur/serve eder
4. Fail-closed: model validation başarısız → uygulama başlamaz
5. NORM türetilir: `1 - max(MI, STTC, CD, HYP)` — doğrudan tahmin edilmez
6. `frontend-legacy/` **dokunulmaz** (arşiv)
7. Ensemble production default: **0.15 CNN / 0.85 XGB** (`artifacts/thresholds_superclass.json`)

### Production akış diyagramı

```
Frontend (cardioguard.ts)
  → POST /predict/superclass
  → main.py: parse_ecg_file → validate_ecg_signal
  → run_in_threadpool(pipeline_predict)
  → run_inference_superclass.predict()
      → core_predict() [normalize → CNN → XGB → ensemble → thresholds → consistency → localization]
      → [explain] xai/pipeline.explain()
  → api_mapper + airesult_mapper
  → JSON response + /runs/{id}/ artifact URLs
```

---

## 2. Faz 0 Bulguları (önceden bilinen — doğrulanacak)

Her bulgu **doğrulama agent'ı** tarafından kod satırı ile teyit edilmeli. **Validation turu tamamlandı (2026-06-14)** — detay: [VALIDATION_REPORT.md](./VALIDATION_REPORT.md).

### P0 — Kritik

| ID | Bulgu | Konum | İlgili WP | Durum |
|----|-------|-------|-----------|-------|
| F-P0-01 | Localization normalization split: superclass path `apply_superclass_normalization()` uygular; standalone `/predict/mi-localization` sadece channel-first | `core_predict` L197–198 vs `run_inference_localization.py` L51 | WP-01, WP-03 | DOĞRULANDI |
| F-P0-02 | Docker UI kırık: `frontend/dist/client/` içinde `index.html` yok; FastAPI sadece `/assets` mount eder; SSR server çalışmıyor | `main.py` L838–844, `Dockerfile`, `docker-compose.yml` | WP-18 | DOĞRULANDI |
| F-P0-03 | CI workflow untracked veya remote'da aktif değil | `.github/workflows/ci.yml` | WP-06 | DOĞRULANDI |
| F-P0-04 | Eğitim ↔ inference normalizasyon uyumsuzluğu (WP-01 kısmen düzeltildi — regression test ile teyit gerekli) | `signal.py`, `run_inference_superclass.py` | WP-01 | KISMEN |

### P1 — Yüksek

| ID | Bulgu | Konum | İlgili WP | Durum |
|----|-------|-------|-----------|-------|
| F-P1-01 | Dual superclass loader: API `load_model_safe()` → `ECGCNN`; CLI `load_cnn_model()` → `MultiLabelECGCNN` | `main.py` AppState, `run_inference_superclass.py` | — | DOĞRULANDI |
| F-P1-02 | `feature_schema.json` yükleniyor ama `validate_feature_schema()` inference'da çağrılmıyor | `model_loader.py`, `core_predict` | WP-04 | DOĞRULANDI |
| F-P1-03 | `main.py` ~100 satır model loading — mimari kural ihlali riski | `main.py` AppState.load_models L231–337 | — | KISMEN |
| F-P1-04 | Duplicate signal I/O: `parse_ecg_file()` vs pipeline `load_ecg_signal()` | `main.py`, inference scripts | — | DOĞRULANDI |
| F-P1-05 | `should_run_localization()` tanımlı ama `core_predict` içinde kullanılmıyor | `consistency_guard.py` L107–128 | — | DOĞRULANDI |
| F-P1-06 | Frontend `airesult` gönderiliyor (`full=true`) ama okunmuyor | `cardioguard.ts`, `mapResultToContext.ts` | WP-11 | DOĞRULANDI |
| F-P1-07 | Frontend `versions` (model_hash, threshold_hash) map'te drop ediliyor | `mapResultToContext.ts` | WP-11 | DOĞRULANDI |
| F-P1-08 | Eval script kırık importlar | `run_comprehensive_test.py`, `generate_validation_predictions.py`, `generate_xai_report.py` | WP-14 | DOĞRULANDI |
| F-P1-09 | MI localization response parity eksik (latency_ms, glossary, versions, full/airesult) | `main.py` predict_mi_localization | — | DOĞRULANDI |
| F-P1-10 | Ensemble weight default drift: eval scriptler 0.5, production 0.15 | `optimize_thresholds.py`, `evaluate_ensemble.py` | WP-02, WP-17 | DOĞRULANDI |

### P2 — Orta

| ID | Bulgu | Konum | İlgili WP | Durum |
|----|-------|-------|-----------|-------|
| F-P2-01 | XAI: 3 farklı manifest writer (API pipeline, XAIReporter batch, localization) | `pipeline.py`, `reporting.py`, `run_inference_localization.py` | WP-15 | DOĞRULANDI |
| F-P2-02 | `extract_highlights()` batch format bekliyor; API path `cards.jsonl` yazmıyor | `contracts/artifacts.py` | WP-15 | DOĞRULANDI |
| F-P2-03 | Sanity status vocabulary mismatch (RELIABLE vs PASS/FAIL) | `sanity.py` vs `artifacts.py` | WP-15 | DOĞRULANDI |
| F-P2-04 | Frontend test yüzeysel: 2 dosya, 8 vitest case | `frontend/src/lib/*.test.ts` | WP-16 | DOĞRULANDI |
| F-P2-05 | `ensure_channel_first()` 5+ yerde kopyalı | `signal.py` + inference scripts | — | DOĞRULANDI |
| F-P2-06 | `validate_ecg_signal()` API path'te iki kez çağrılıyor | `parse_ecg_file` + `core_predict` | — | DOĞRULANDI |
| F-P2-07 | Welcome "Sistem Durumu" statik yeşil; `/ready` kontrolü yok | `index.tsx` vs legacy `SystemStatus` | WP-09 | DOĞRULANDI |
| F-P2-08 | `TechnicalDetails` fake metadata (model_hash sessionId slice) | `TechnicalDetails.tsx` | WP-11 | DOĞRULANDI |
| F-P2-09 | Legacy binary path tamamen farklı stack (XGB, XAI, norm) | `run_inference_binary.py` | — | DOĞRULANDI |
| F-P2-10 | `combined.py` / `CombinedExplainer` referansları var, dosya yok | docs, `reporting.py` docstring | WP-17 | DOĞRULANDI |

### Zaten düzeltilmiş (doğrulama agent teyit etmeli)

| ID | Bulgu | Beklenen durum | İlgili WP | Durum |
|----|-------|----------------|-----------|-------|
| F-FIX-01 | `@app.on_event("startup")` deprecated | `lifespan` kullanılıyor | — | GÜNCELLENDİ |
| F-FIX-02 | Sync ML blocking event loop | `run_in_threadpool` kullanılıyor | WP-05 | GÜNCELLENDİ |
| F-FIX-03 | CORS hardcoded | `CORS_ORIGINS` env | WP-07 | GÜNCELLENDİ |
| F-FIX-04 | Debug endpoint açık | `ENABLE_DEBUG_ENDPOINTS` gate | WP-07 | GÜNCELLENDİ |
| F-FIX-05 | Ensemble default drift API/frontend | 0.15 SSOT | WP-02 | GÜNCELLENDİ |
| F-FIX-06 | Demo modu mock inference zorluyor | `analysisMode.ts` ayrımı | WP-10 | GÜNCELLENDİ |
| F-FIX-07 | Fail-closed startup warn-only | RuntimeError on missing required | WP-08 | GÜNCELLENDİ |

---

## 3. Agent Tanımları (14 audit agent)

Her agent **readonly** çalışır: kod okur, bulgular üretir, **kod değiştirmez**.

### Standart agent prompt şablonu

```
GÖREV: [A-XX adı] — CardioGuard-AI audit
KAPSAM: [dosya listesi]
KURALLAR: docs/agents/ORCHESTRATION_PLAN.md §1 mimari kurallar
YAP:
1. Kodu oku, akışı çiz
2. ORCHESTRATION_PLAN.md'deki [ilgili F-* ID'leri] doğrula veya reddet (satır referansı şart)
3. Yeni bulgular varsa F-NEW-XX olarak ekle
4. Her bulgu: öncelik, dosya:satır, risk, önerilen fix, test önerisi, WP eşlemesi
YAPMA: Kod değiştirme, commit yapma
ÇIKTI: Markdown rapor (max 500 satır)
```

---

### Faz 1 — Temel katmanlar (paralel)

#### A-01: Signal & Normalization

| Alan | Detay |
|------|-------|
| **Dosyalar** | `src/utils/signal.py`, `tests/test_signal_normalization.py`, `tests/test_ecg_validation.py` |
| **Doğrulayacak bulgular** | F-P0-01, F-P0-04, F-P2-05, F-P2-06, F-FIX-01 |
| **Odak sorular** | Z-score stats yükleme yolu? Fallback JSON? `ensure_channel_first` kopyaları? API vs pipeline validate sırası? |
| **Bağımlılık** | — |

#### A-02: Model Loader & Checkpoints

| Alan | Detay |
|------|-------|
| **Dosyalar** | `src/utils/model_loader.py`, `src/utils/checkpoint_validation.py`, `src/backend/main.py` (AppState), `tests/test_checkpoint_validation.py`, `tests/test_startup_failclosed.py` |
| **Doğrulayacak bulgular** | F-P1-01, F-P1-02, F-P1-03, F-FIX-07 |
| **Odak sorular** | Dual loader riski? `weights_only`? feature_schema inference'da validate? Fail-closed tam mı? |
| **Bağımlılık** | — |

#### A-03: Data Layer

| Alan | Detay |
|------|-------|
| **Dosyalar** | `src/data/*`, `tests/test_data.py`, `src/config.py` (PTBXLConfig) |
| **Doğrulayacak bulgular** | NORM türetimi, split leakage, MI fingerprint |
| **Odak sorular** | Label tutarlılığı? `min_likelihood` default drift (50 vs 0)? Patient-level split? |
| **Bağımlılık** | — |

---

### Faz 2 — Inference çekirdeği

#### A-04: Inference Core

| Alan | Detay |
|------|-------|
| **Dosyalar** | `src/pipeline/inference/run_inference_superclass.py` (özellikle `core_predict`, `predict`, `get_primary_label`, `load_thresholds`) |
| **Doğrulayacak bulgular** | F-P0-04, F-P1-02, F-P1-05, F-FIX-05 |
| **Odak sorular** | Ensemble formül SSOT? Threshold loading? NORM derivation? XGB fallback? |
| **Bağımlılık** | A-01, A-02 |

#### A-05: Localization Path

| Alan | Detay |
|------|-------|
| **Dosyalar** | `run_inference_localization.py`, `consistency_guard.py`, `src/data/mi_localization.py`, `main.py` (mi-localization endpoint) |
| **Doğrulayacak bulgular** | F-P0-01, F-P1-05, F-P1-09 |
| **Odak sorular** | Normalization split? Embedded vs standalone? Response parity? |
| **Bağımlılık** | A-01, A-04 |

#### A-06: Legacy Binary Path

| Alan | Detay |
|------|-------|
| **Dosyalar** | `run_inference_binary.py`, `src/xai/shap_xgb.py`, `src/xai/summary.py` |
| **Doğrulayacak bulgular** | F-P2-09 |
| **Odak sorular** | Consistency guard binary model drift? API'dan çağrılmıyor mu? |
| **Bağımlılık** | A-02 |

---

### Faz 3 — Backend & contracts

#### A-07: Backend API Gateway

| Alan | Detay |
|------|-------|
| **Dosyalar** | `src/backend/main.py` (tümü), `tests/test_api.py` |
| **Doğrulayacak bulgular** | F-P0-02, F-P1-03, F-P1-04, F-P1-09, F-FIX-02, F-FIX-03, F-FIX-04 |
| **Odak sorular** | Endpoint listesi tam? ML kodu kaldı mı? Threadpool? Static mount? Debug gate? |
| **Bağımlılık** | A-04 |

#### A-08: Contracts Layer

| Alan | Detay |
|------|-------|
| **Dosyalar** | `src/contracts/*`, `tests/test_airesult_mapper.py`, `tests/test_frontend_contract_coverage.py`, `tests/test_artifacts.py`, `tests/test_explanation_summary.py` |
| **Doğrulayacak bulgular** | F-P1-06, F-P1-07, F-P2-02, F-P2-03 |
| **Odak sorular** | Dual mapper drift? AIResult vs API response? Artifact discovery tutarlı? |
| **Bağımlılık** | A-07 |

---

### Faz 4 — XAI

#### A-09: XAI Production Path

| Alan | Detay |
|------|-------|
| **Dosyalar** | `src/xai/pipeline.py`, `gradcam.py`, `shap_ovr.py`, `unified.py`, `sanity.py`, `visualize.py`, `tests/test_gradcam.py`, `tests/test_xai_sanity.py`, `tests/test_unified_gradcam.py` |
| **Doğrulayacak bulgular** | F-P2-01, F-P2-02, F-P2-03 |
| **Odak sorular** | `explain()` E2E? Manifest schema? Sanity default? Test coverage gaps? |
| **Bağımlılık** | A-04 |

#### A-10: XAI Legacy & Batch

| Alan | Detay |
|------|-------|
| **Dosyalar** | `src/xai/reporting.py`, `src/pipeline/xai/*`, `shap_xgb.py`, `summary.py`, `tests/test_xai_visualization.py` |
| **Doğrulayacak bulgular** | F-P1-08, F-P2-10 |
| **Odak sorular** | Kırık importlar? Batch vs API manifest? Legacy stack drift? |
| **Bağımlılık** | A-09 |

---

### Faz 5 — Frontend

#### A-11: Frontend UX & Flow

| Alan | Detay |
|------|-------|
| **Dosyalar** | `frontend/src/routes/*`, `hooks/useAnalysisSession.ts`, `components/evidence/*`, `components/chat/*` |
| **Doğrulayacak bulgular** | F-P2-07, F-P2-08, F-FIX-06 |
| **Odak sorular** | Demo vs live? Session restore? Health UX? ECG mock vs real signal? |
| **Bağımlılık** | A-08 |

#### A-12: Frontend Contracts

| Alan | Detay |
|------|-------|
| **Dosyalar** | `lib/api/cardioguard.ts`, `mapResultToContext.ts`, `types.ts`, `analysisMode.ts`, `*.test.ts` |
| **Doğrulayacak bulgular** | F-P1-06, F-P1-07, F-P2-04, F-P2-08 |
| **Odak sorular** | Zod yok? Ensemble inversion? Timeout yok? Test gaps? |
| **Bağımlılık** | A-08 |

---

### Faz 6 — Infra & offline

#### A-13: Docker & CI

| Alan | Detay |
|------|-------|
| **Dosyalar** | `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.github/workflows/ci.yml`, `frontend/vite.config.ts`, `frontend/package.json` |
| **Doğrulayacak bulgular** | F-P0-02, F-P0-03, F-P2-04 |
| **Odak sorular** | UI Docker'da açılıyor mu? CI tracked? test_data exclude? Python 3.10 vs 3.12? |
| **Bağımlılık** | — |

#### A-14: Training & Eval

| Alan | Detay |
|------|-------|
| **Dosyalar** | `src/pipeline/training/*`, `src/pipeline/evaluation/*`, `src/pipeline/features/*`, `src/models/*`, `artifacts/thresholds_superclass.json` |
| **Doğrulayacak bulgular** | F-P0-01, F-P1-08, F-P1-10, F-P2-09 |
| **Odak sorular** | Import paths? ensemble_weight defaults? Localization train norm? Feature npz eksik? |
| **Bağımlılık** | A-01, A-03 |

---

### Faz 7 — Sentez

#### A-15: Sentez & WP Matrisi (orchestrator)

| Alan | Detay |
|------|-------|
| **Girdi** | A-01 … A-14 raporları |
| **Çıktı** | Güncellenmiş bulgu tablosu (DOĞRULANDI / REDDEDİLDİ / KISMEN), WP öncelik sırası, `technical_debt_inventory.md` diff önerisi |
| **Bağımlılık** | Tüm fazlar |

---

## 4. Orkestrasyon sırası

```
Faz 0  [TAMAMLANDI] Proje haritası + ön bulgular
         │
Faz 1  A-01 + A-02 + A-03          (paralel, 3 agent)
         │
Faz 2  A-04 → A-05 + A-06          (A-04 önce; 5+6 paralel)
         │
Faz 3  A-07 → A-08
         │
Faz 4  A-09 → A-10
         │
Faz 5  A-11 + A-12                 (paralel)
         │
Faz 6  A-13 + A-14                 (Faz 1 ile paralel başlayabilir)
         │
Faz 7  A-15 Sentez
         │
Faz 8  VALIDATION_PROMPT.md ile Plan modu doğrulama turu
```

---

## 5. WP eşleme tablosu

| WP | Konu | Agent |
|----|------|-------|
| WP-01 | Inference normalizasyon | A-01, A-04 |
| WP-02 | Ensemble SSOT | A-04, A-14 |
| WP-03 | ECG validation | A-01 |
| WP-04 | Feature schema validation | A-02, A-04 |
| WP-05 | Async threadpool | A-07 |
| WP-06 | GitHub Actions CI | A-13 |
| WP-07 | CORS + debug | A-07 |
| WP-08 | Fail-closed startup | A-02, A-07 |
| WP-09 | Frontend health UX | A-11 |
| WP-10 | Demo vs live | A-11 |
| WP-11 | Contract alignment | A-08, A-12 |
| WP-12 | Zod validation | A-12 |
| WP-13 | LLM timeout UX | A-11 |
| WP-14 | Eval import fix | A-14 |
| WP-15 | XAI sanity baseline | A-09, A-10 |
| WP-16 | Frontend test expansion | A-12 |
| WP-17 | Documentation sync | A-15 |
| WP-18 | Docker frontend (HITL) | A-13 |

Tam WP spec: `docs/agents/work_packages/README.md`

---

## 6. Doğrulama komutları (agent acceptance)

```bash
# Backend
pytest tests/ -q --ignore=tests/test_data.py

# Frontend
cd frontend && npm test && npx tsc --noEmit

# Docker smoke (WP-18 öncesi beklenen: /health OK, / UI kırık)
docker-compose up --build
curl -s http://localhost:8000/health
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/
```

---

## 7. İlgili dosyalar

| Dosya | Amaç |
|-------|------|
| `docs/agents/VALIDATION_REPORT.md` | Faz V1–V7 doğrulama raporu (2026-06-14) |
| `docs/agents/VALIDATION_PROMPT.md` | Plan modu doğrulama prompt'u |
| `docs/agents/work_packages/README.md` | WP indeksi ve bağımlılıklar |
| `docs/agents/work_packages/REVIEW_PROTOCOL.md` | Implementasyon sonrası review |
| `docs/technical_debt_inventory.md` | Teknik borç envanteri |
| `CLAUDE.md` | Mimari kurallar ve aktif dosya haritası |
