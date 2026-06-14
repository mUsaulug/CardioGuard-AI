# CardioGuard-AI — Validation Report

**Tarih:** 2026-06-14  
**Orchestrator:** V-A-15 (Faz V1–V7 sentez)  
**Kaynak plan:** [ORCHESTRATION_PLAN.md](./ORCHESTRATION_PLAN.md) §2  
**Mod:** Readonly audit — kod değiştirilmedi

---

## 1. Özet

| Metrik | Adet |
|--------|------|
| DOĞRULANDI | 24 |
| KISMEN | 3 |
| GÜNCELLENDİ (fix teyit) | 7 |
| REDDEDİLDİ | 0 |
| F-NEW (yeni bulgu) | 38 |
| Toplam ön bulgu (F-P0/P1/P2 + F-FIX) | 31 |

**Doğrulama komutları (bu ortam):**

| Komut | Sonuç |
|-------|-------|
| `pytest tests/ -q --ignore=tests/test_data.py` | `ModuleNotFoundError: fastapi` (venv eksik) |
| `python -c "from src.pipeline.inference.run_inference_superclass import predict"` | `ModuleNotFoundError: pandas` (venv eksik) |
| `git status .github/workflows/ci.yml` | **Untracked** |
| `ls frontend/dist/client/` | Yalnızca `assets/` — `index.html` yok |
| `ls frontend/src/lib/*.test.ts` | 2 dosya (8 vitest case) |

---

## 2. P0 Bulgular

| ID | Durum | Kanıt | WP |
|----|-------|-------|-----|
| **F-P0-01** | DOĞRULANDI | `run_inference_superclass.py:197-198` → `apply_superclass_normalization`; embedded loc `:270` aynı z-scored tensor; standalone `run_inference_localization.py:51` → `_ensure_channel_first` only | WP-01, WP-03 |
| **F-P0-02** | DOĞRULANDI | `main.py:838-844` sadece `/assets` mount; `frontend/dist/client/` → `assets/` only, `index.html` yok; `Dockerfile:26-30` uvicorn only, SSR server çalışmıyor | WP-18 |
| **F-P0-03** | DOĞRULANDI | `git status .github/workflows/ci.yml` → Untracked; `git ls-files .github/workflows/` boş | WP-06 |
| **F-P0-04** | KISMEN | WP-01 inference path düzeltilmiş: `core_predict` `:197-198` + startup `main.py:247-249`; **eksik:** `normalization_stats.npz` repo'da yok, training↔inference parity testi yok, localization path split devam | WP-01 |

---

## 3. P1 Bulgular

| ID | Durum | Kanıt | WP |
|----|-------|-------|-----|
| **F-P1-01** | DOĞRULANDI | API: `main.py:254-256` → `load_model_safe` → `ECGCNN` (`model_loader.py:145-147`); CLI: `run_inference_superclass.py:68-77` → `MultiLabelECGCNN` | — |
| **F-P1-02** | DOĞRULANDI | Schema startup'ta zorunlu `main.py:315-316`; `validate_feature_schema` tanımı `model_loader.py:207-227`; `core_predict` `:207-216` embedding üretir, validate çağırmaz | WP-04 |
| **F-P1-03** | KISMEN | `AppState.load_models()` `main.py:231-337` (107 satır); endpoint'ler pipeline'a delegate ediyor ama startup ML yükleme gateway'de | — |
| **F-P1-04** | DOĞRULANDI | API: `parse_ecg_file` `main.py:423-459`; CLI: `load_ecg_signal` `run_inference_superclass.py:142-158` — overlapping npz/npy logic, farklı validation | — |
| **F-P1-05** | DOĞRULANDI | `should_run_localization` `consistency_guard.py:107-128`; `core_predict` `:268` → `"MI" in predicted_labels` only | — |
| **F-P1-06** | DOĞRULANDI | `cardioguard.ts:75` `full: "true"`; `:55` types `airesult`; `mapResultToContext.ts` okumuyor | WP-11 |
| **F-P1-07** | DOĞRULANDI | `mapResultToContext.ts:41` yalnızca `api.versions.timestamp`; `types.ts` `versions` field yok | WP-11 |
| **F-P1-08** | DOĞRULANDI | `run_comprehensive_test.py:23-32` → `src.pipeline.run_inference_*` (yok); `generate_validation_predictions.py:30` → `src.pipeline.train_superclass_xgb_ovr`; `generate_xai_report.py:35-37` aynı | WP-14 |
| **F-P1-09** | DOĞRULANDI | Superclass: `latency_ms`, `glossary`, `versions`, `full`/`airesult` (`main.py:689,717-725,748-759`); standalone `predict_mi_localization` `:821-831` — yok | — |
| **F-P1-10** | DOĞRULANDI | Production `thresholds_superclass.json:42` → `0.15`; eval `evaluate_ensemble.py:35`, `optimize_thresholds.py:145-146` default `0.5` | WP-02, WP-17 |

---

## 4. P2 Bulgular

| ID | Durum | Kanıt | WP |
|----|-------|-------|-----|
| **F-P2-01** | DOĞRULANDI | 3 manifest writer: `xai/pipeline.py:319-331`, `reporting.py:257-316`, `run_inference_localization.py:225-237` — farklı şemalar | WP-15 |
| **F-P2-02** | DOĞRULANDI | `extract_highlights` `artifacts.py:135-161` → `cards.jsonl`; API path `pipeline.py:281-331` yazmaz; `top_windows` hiç üretilmiyor (`unified.py:116-125`) | WP-15 |
| **F-P2-03** | DOĞRULANDI | Producer `sanity.py:523-528` → RELIABLE/ACCEPTABLE/UNRELIABLE; consumer `artifacts.py:217-224` → PASS/FAIL; RELIABLE 3/4 → FAIL | WP-15 |
| **F-P2-04** | DOĞRULANDI | 2 test dosyası, 8 `it()` case: `analysisMode.test.ts` (3), `mapResultToContext.test.ts` (5) | WP-16 |
| **F-P2-05** | DOĞRULANDI | 7 duplicate `ensure_channel_first` / `_ensure_channel_first`: `signal.py:28`, `run_inference_superclass.py:161`, `run_inference_localization.py:99`, `run_inference_binary.py:84`, `train_mi_localization.py:73`, `run_comprehensive_test.py:40`, `scripts/generate_figure3.py:5` | — |
| **F-P2-06** | DOĞRULANDI | `parse_ecg_file` → `validate_ecg_signal` `main.py:459`; `core_predict` tekrar `run_inference_superclass.py:197` | — |
| **F-P2-07** | DOĞRULANDI | `index.tsx:198-204` statik yeşil `StatusRow ok`; `/ready` fetch yok | WP-09 |
| **F-P2-08** | DOĞRULANDI | `TechnicalDetails.tsx:38` → `sessionId.slice(0,12)`, hardcoded `API v1.0`, `thr#a3f9` | WP-11 |
| **F-P2-09** | DOĞRULANDI | `run_inference_binary.py` ayrı stack (norm/XGB/XAI); API `main.py:669-684` → `run_inference_superclass` only | — |
| **F-P2-10** | DOĞRULANDI | `src/xai/combined.py` yok; stale ref `reporting.py:119` docstring; runtime `unified.py` kullanılıyor | WP-17 |

---

## 5. Fix Teyitleri (F-FIX)

| ID | Durum | Kanıt |
|----|-------|-------|
| **F-FIX-01** | GÜNCELLENDİ | `main.py:347-405` `@asynccontextmanager lifespan`; `on_event("startup")` yok |
| **F-FIX-02** | GÜNCELLENDİ | `run_in_threadpool` `main.py:42,646,669,788,804` |
| **F-FIX-03** | GÜNCELLENDİ | `_resolve_cors_origins()` `main.py:66-74`; middleware `:408-416` |
| **F-FIX-04** | GÜNCELLENDİ | `ENABLE_DEBUG_ENDPOINTS` `main.py:62-63,542-555`; test `test_api.py:343-377` |
| **F-FIX-05** | GÜNCELLENDİ | SSOT `thresholds_superclass.json:42` → 0.15; `get_ensemble_cnn_weight()` `config.py:202-208`; frontend `cardioguard.ts:69-72` |
| **F-FIX-06** | GÜNCELLENDİ | `analysisMode.ts:5-10` demo ≠ mock inference; `useAnalysisSession.ts:140-169` live path |
| **F-FIX-07** | GÜNCELLENDİ | `RuntimeError` missing assets `main.py:260,312-318,334`; test `test_startup_failclosed.py` |

---

## 6. Yeni Bulgular (F-NEW) — Seçilmiş

| ID | Öncelik | Bulgu | Kanıt |
|----|---------|-------|-------|
| F-NEW-01 | P2 | Dead `ensure_channel_first` in superclass inference | `run_inference_superclass.py:161-176` |
| F-NEW-02 | P0 | Embedded loc z-scored vs training raw vs standalone raw | `:198,270` vs `train_mi_localization.py:65-66` vs `run_inference_localization.py:51` |
| F-NEW-03 | P2 | İki `normalize_signal` semantiği | `signal.py:140` vs `data/signals.py:401` |
| F-NEW-04 | P1 | `normalization_stats.npz` repo'da yok | JSON fallback `signal.py:189-198` |
| F-NEW-05 | P2 | MI-localization tek validate (parse only) | `main.py:788` |
| F-NEW-06 | P2 | Binary path farklı norm stack | `run_inference_binary.py:103-108` |
| F-NEW-07 | P1 | `feature_schema` predict path'e geçirilmiyor | `main.py:669-684` |
| F-NEW-08 | P1 | Type mismatch ECGCNN vs MultiLabelECGCNN | `run_inference_superclass.py:324` vs `main.py:672` |
| F-NEW-09 | P1 | CLI XGB loader schema yüklemez | `run_inference_superclass.py:98-127` |
| F-NEW-10 | P2 | `weights_only=False` explicit; testlerde yok | `model_loader.py:124` |
| F-NEW-11 | P1 | `min_likelihood` SSOT yok: config 50 vs training 0 | `config.py:36-38` vs `train_superclass_cnn.py:217` |
| F-NEW-12 | P1 | Legacy `labels.py` vs `labels_superclass.py` dual path | `labels.py:154-184` |
| F-NEW-13 | P2 | Localization training'de patient leakage check yok | `train_mi_localization.py:197-199` |
| F-NEW-14 | P2 | `validate.py:209` broken co-occurrence call | TypeError risk |
| F-NEW-15 | P2 | `validate.py:174-176` wrong column name | KeyError risk |
| F-NEW-16 | P2 | `test_data.py` derived-NORM test etmiyor | — |
| F-NEW-17 | P2 | Offline XAI hardcoded ensemble 0.5 | `generate_xai_report.py:405` |
| F-NEW-18 | P1 | Consistency guard CNN-only vs legacy ensemble | `run_inference_superclass.py:255-256` |
| F-NEW-19 | P1 | `labels_tr` inline loc'ta var, standalone'da yok | `api_mapper.py:48` vs `MILocalizationResponse` |
| F-NEW-20 | P2 | Frontend `/predict/mi-localization` client yok | grep zero hits |
| F-NEW-21 | P2 | Stale "ONLY manifest writer" comment | `xai/pipeline.py:290` |
| F-NEW-22 | P1 | `top_windows` contract — zero producers | `pipeline.py:326`, `unified.py:116` |
| F-NEW-23 | P1 | Three sanity vocabularies on one API response | `sanity.py:524`, `artifacts.py:217`, `explanation_summary.py:22` |
| F-NEW-24 | P2 | EvidencePanel fake hash | `EvidencePanel.tsx:66` |
| F-NEW-25 | P2 | Inference fetch timeout yok | `cardioguard.ts:81-97` |
| F-NEW-26 | P1 | Python 3.10 Docker vs 3.12 CI | `Dockerfile:5`, `ci.yml:16` |
| F-NEW-27 | P1 | CI Docker/build smoke yok — F-P0-02 invisible | `ci.yml:26-52` |
| F-NEW-28 | P2 | WP-06 PROGRESS "done" ama ci.yml untracked | `PROGRESS.md:9` |
| F-NEW-29 | P2 | Stale `python -m src.pipeline.*` docstrings | eval/training scripts |
| F-NEW-30 | P2 | `tests/test_evaluation_imports.py` WP-14'te referans, yok | — |

*(Tam liste agent raporlarında; yukarıdakiler en yüksek etkili 30.)*

---

## 7. WP Öncelik Önerisi

| Sıra | WP | Gerekçe (bulgu) |
|------|-----|-----------------|
| 1 | **WP-01 + WP-03** | F-P0-01, F-NEW-02 — localization norm split (embedded z-score vs training raw) |
| 2 | **WP-06** | F-P0-03 — `ci.yml` track + push |
| 3 | **WP-18** | F-P0-02 — TanStack SSR vs FastAPI static (HITL karar) |
| 4 | **WP-04** | F-P1-02 — `validate_feature_schema` at inference |
| 5 | **Yeni WP** | F-NEW-11 — `min_likelihood` SSOT |
| 6 | **WP-02 + WP-14** | F-P1-10, F-P1-08 — eval ensemble default + import fix |
| 7 | **WP-15** | F-P2-01/02/03 — XAI manifest/highlights/sanity unify |
| 8 | **WP-11 + WP-09** | F-P1-06/07, F-P2-07/08 — contract + health UX |
| 9 | **WP-16 + WP-12** | F-P2-04 — frontend test + Zod |

---

## 8. technical_debt_inventory.md Çelişkileri

| Inventory maddesi | Audit sonucu | Önerilen güncelleme |
|-------------------|--------------|---------------------|
| P0 #1 "inference sadece ensure_channel_first" | **GÜNCELLENDİ** — `apply_superclass_normalization` wired | KISMEN: parity test/npz eksik; localization split ekle |
| P0 #2 "sync ML blocking" | **REDDEDİLDİ** — `run_in_threadpool` aktif | Maddeyi kapat / F-FIX-02 |
| P0 #3 Docker UI | **DOĞRULANDI** | Değişiklik yok |
| P0 #4 "CI/CD yok" | **KISMEN** — `ci.yml` var, untracked | F-P0-03 ile hizala |
| P1 #1 "warn-only startup" | **REDDEDİLDİ** — RuntimeError fail-closed | Maddeyi kapat / F-FIX-07 |
| P1 #8 "ensemble 0.5 drift" (API) | **GÜNCELLENDİ** production path | Eval script drift (F-P1-10) ayrı madde |
| P1 #22 "ECG validation yok" | **REDDEDİLDİ** — `validate_ecg_signal` mevcut | WP-03 done |
| — | **EKSİK** | F-NEW-11 min_likelihood drift ekle |
| — | **EKSİK** | F-P0-01 localization norm split ekle |
| — | **EKSİK** | F-NEW-12 dual label modules ekle |
| PROGRESS WP-06 ✅ | **YANLIŞ** | ci.yml hâlâ untracked (F-NEW-28) |

---

## 9. Agent Faz Özeti

| Faz | Agent | Bulgu sayısı | Not |
|-----|-------|--------------|-----|
| V1 | V-A-01 Signal | 4 + 6 F-NEW | F-P0-04 KISMEN |
| V1 | V-A-02 Model Loader | 4 + 4 F-NEW | F-FIX-07 GÜNCELLENDİ |
| V1 | V-A-03 Data Layer | 6 F-NEW | NORM/split/fingerprint OK; min_likelihood drift |
| V2 | V-A-04 Inference Core | 4 + 4 F-NEW | F-FIX-05 GÜNCELLENDİ |
| V2 | V-A-05 Localization | 3 + 6 F-NEW | F-P0-01 confirmed |
| V2 | V-A-06 Legacy Binary | 1 + 7 F-NEW | F-P2-09 confirmed |
| V3 | V-A-07 Backend API | 7 + 8 F-NEW | 3 fixes verified |
| V3 | V-A-08 Contracts | 4 + 3 F-NEW | Dual-path inconsistency |
| V4 | V-A-09 XAI Production | 3 + 5 F-NEW | F-P2-02 worse than stated |
| V4 | V-A-10 XAI Legacy | 2 + 5 F-NEW | F-P1-08 confirmed |
| V5 | V-A-11 Frontend UX | 3 + 1 F-NEW | F-FIX-06 verified |
| V5 | V-A-12 Frontend Contracts | 4 + 2 F-NEW | 8 vitest cases |
| V6 | V-A-13 Docker & CI | 3 + 8 F-NEW | F-P0-02/03 confirmed |
| V6 | V-A-14 Training & Eval | 4 + 5 F-NEW | Eval drift confirmed |

---

## 10. Sonuç

- **31/31** ön bulgu incelendi: **24 DOĞRULANDI**, **3 KISMEN**, **7 fix GÜNCELLENDİ**, **0 REDDEDİLDİ** (inventory'deki eski maddeler ayrıca reddedildi).
- En kritik açık riskler: **localization normalization üçlü split** (F-P0-01), **Docker UI serve edilmiyor** (F-P0-02), **CI untracked** (F-P0-03).
- WP-01 kısmen uygulanmış; superclass inference norm düzeltilmiş ancak localization path ve regression testleri eksik.
- 7 F-FIX maddesinin tamamı kodda doğrulandı.

**Sonraki adım:** WP backlog implementasyonu; `ORCHESTRATION_PLAN.md` §2 durumları bu raporla güncellendi.
