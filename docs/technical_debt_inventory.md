# CardioGuard — Teknik Borç Envanteri

**Tarih:** 2026-06-14 (validation sync)  
**Önceki tarama:** 2026-06-09  
**Kapsam:** Tam repo + `docs/agents/VALIDATION_REPORT.md` doğrulama  
**Yöntem:** Readonly audit + implementasyon turu (2026-06-14)

---

## Validation sync özeti (2026-06-14)

| Durum | Açıklama |
|-------|----------|
| ✅ Düzeltildi | Superclass inference z-score, `run_in_threadpool`, fail-closed startup, CORS/debug gate, ensemble 0.15 SSOT, demo/live ayrımı, feature_schema inference validate, eval imports, standalone localization norm, Docker multi-stage, SPA routes, frontend health UX, Zod API parse, MI-localization response parity (versions/latency/glossary) |
| ⚠️ Kısmen | Superclass norm regression / `normalization_stats.npz` repo'da yok; localization model eğitimi raw sinyal (retrain önerilir); dual ECGCNN/MultiLabelECGCNN loader |
| 📁 Dosyada | `.github/workflows/ci.yml` — commit bekliyor |
| 🔜 Açık | Dual loader birleştirme, signal_io konsolidasyonu, XAI manifest unify, WP-16+ test genişletme, OpenRouter backend proxy |

Detay: [`docs/agents/VALIDATION_REPORT.md`](agents/VALIDATION_REPORT.md)

---

## Özet sayılar (güncellenmiş tahmin)

| Önem | Adet (yaklaşık) | Ana temalar |
|------|-----------------|-------------|
| **Kritik** | 4 | Inference normalizasyonu, async blocking, Docker frontend, CI yok |
| **Yüksek** | ~35 | Ensemble drift, güvenlik, contract drift, kırık eval importları |
| **Orta** | ~45 | Çift loader, XAI gaps, test boşlukları, docs stale |
| **Düşük** | ~25 | Hardcoded path, doc typos, bundle şişkinliği |

---

## Zaten düzeltilmiş (CLAUDE.md güncel değil)

Aşağıdakiler **kodda yapılmış**; `CLAUDE.md` "Kalan Sorunlar" bölümü eski:

| Konu | Durum |
|------|--------|
| `@app.on_event("startup")` | ✅ `lifespan` kullanılıyor (`src/backend/main.py`) |
| `datetime.utcnow()` | ✅ Production `src/` yolunda yok |
| Coherence placeholder 0.85 | ✅ Gerçek hesaplama |
| CORS env | ✅ `CORS_ORIGINS` |
| Consistency Guard | ✅ Pipeline'da aktif (`run_inference_superclass.py`) |

---

## P0 — Kritik

### 1. Eğitim ↔ inference normalizasyon (superclass)
- **Durum:** ✅ Inference path düzeltildi (`apply_superclass_normalization` in `core_predict`)
- **Kalan:** `normalization_stats.npz` artifact repo'da yok (JSON fallback); localization model hâlâ raw ile eğitildi — inference z-score kullanıyor

### 2. Sync ML inference async route içinde
- **Durum:** ✅ `run_in_threadpool` aktif (`main.py`)

### 3. Docker frontend serve
- **Durum:** ✅ Multi-stage `Dockerfile` + `index.html` SPA routes in `main.py`
- **Kalan:** TanStack build çıktısında `index.html` varlığı build'e bağlı — `docker-compose up --build` ile doğrula

### 4. CI/CD
- **Durum:** ⚠️ `ci.yml` mevcut, **git'te track edilmedi** — push sonrası aktif

---

## P0 — Eski maddeler (arşiv)

<details>
<summary>2026-06-09 P0 metni (çoğu kapatıldı)</summary>

### 1. Eğitim ↔ inference normalizasyon uyumsuzluğu (ESKİ)
- Inference sadece `ensure_channel_first` — **giderildi**

### 2. Sync ML blocking — **giderildi**

### 3. Docker UI — **kısmen giderildi** (multi-stage + SPA routes)

### 4. CI/CD yok — **ci.yml var**, commit bekliyor

</details>

## P0 — Kritik (ESKİ BLOK KALDIRILDI — yukarıya bak)

---

## P1 — Yüksek öncelik

### Backend & API

| # | Konu | Konum | Fix |
|---|------|-------|-----|
| 1 | Checkpoint `FileNotFoundError` lifespan'de warn-only | `main.py:321-324` | Fail-closed startup |
| 2 | `feature_schema.json` yükleniyor, inference'da validate edilmiyor | `model_loader.py`, `core_predict` | `validate_feature_schema()` çağır |
| 3 | CORS `*` + `allow_credentials=True` | `main.py:339-346` | Prod'da explicit origin |
| 4 | `/debug/client-log` auth yok | `main.py:475-493` | `DEBUG=1` veya dev-only gate |
| 5 | `np.load` pickle riski | `main.py:364,373` | `allow_pickle=False` |
| 6 | Çift superclass loader (`ECGCNN` vs `MultiLabelECGCNN`) | `main.py` vs `run_inference_superclass.py` | Tek loader |
| 7 | Sinyal parse 4 yerde kopyalı | main + 3 inference script | `src/utils/signal_io.py` |
| 8 | Ensemble default drift: API 0.15, CLI 0.5, demo UI 0.85 | config + frontend | `thresholds_superclass.json`'dan oku |
| 9 | `/predict/mi-localization` response parity eksik | `main.py` | `latency_ms`, glossary, versions hizala |
| 10 | Path/config hardcoded | `main.py:48-51,206-238` | `config.py` + env |

### Frontend

| # | Konu | Konum | Fix |
|---|------|-------|-----|
| 11 | Demo modu gerçek dosyada mock'a zorluyor | `useAnalysisSession.ts`, `index.tsx` | Demo ≠ live upload uyarısı |
| 12 | TS `mapResultToContext` vs Python `frontend_context.py` drift | contracts + frontend | Tek contract / OpenAPI |
| 13 | API response runtime validation yok | `cardioguard.ts` | Zod schema |
| 14 | Welcome'de backend health yok | `index.tsx` | `/health` poll |
| 15 | LLM chain ~110s (5×22s), global timeout yok | `openrouter.ts` | Toplam 45s budget |
| 16 | OpenRouter key tarayıcıda açık | client-side fetch | Prod'da backend proxy |
| 17 | Test: sadece 1 vitest dosyası | `mapResultToContext.test.ts` | hooks, openrouter, storage |
| 18 | session restore validation yok | `storage.ts` | Schema validate on load |

### ML / XAI / Data

| # | Konu | Konum | Fix |
|---|------|-------|-----|
| 19 | Eval script kırık importlar | `run_comprehensive_test.py`, `generate_xai_report.py`, `generate_validation_predictions.py` | `src.pipeline.inference.*` path fix |
| 20 | `artifacts/train_baseline.npz` eksik | `sanity.py` | Training'de üret |
| 21 | Sanity check default kapalı | API `sanity_check=false` | Clinical mode'da default açık |
| 22 | ECG shape/amplitude/NaN validation yok | upload handlers | `validate_ecg_signal()` |
| 23 | `input_meta.shape` hardcoded [12,1000] | `airesult_mapper.py` | Gerçek shape |
| 24 | `shap_xgb.py`, `combined.py` referansları | docs + eski scriptler | Deprecate / temizle |
| 25 | XAI runs disk sınırsız büyür | `reports/xai/runs/` | TTL / LRU cleanup |

### Infra & deps

| # | Konu | Konum | Fix |
|---|------|-------|-----|
| 26 | `requirements.txt` unpinned | root | Lockfile / pin |
| 27 | npm + bun çift lockfile | `frontend/` | Tek PM seç |
| 28 | Python 3.10 / 3.11 / 3.12 karışık | Docker, docs, `.venv312` | `.python-version` |
| 29 | `frontend-legacy/` + stale CLAUDE.md paths | repo root | Docs güncelle veya legacy arşivle |
| 30 | `features_out/*.npz` (~11MB) git'te | repo | gitignore + regen script |

---

## P2 — Orta öncelik

### Deprecated / güvenlik

- `torch.load(..., weights_only=False)` — bilinçli mi dokümante et veya `weights_only=True` (model_loader, run_inference_superclass, checkpoint_validation, checkpoints.py)
- Thresholds dosyası yoksa CLI sessiz 0.5 fallback (`run_inference_superclass.py:129-133`)
- Consistency guard hataları yutuluyor → API'de `null` + warning
- `/ready` XGB/binary yokken de `ready=true`
- Localization XAI inline; superclass unified pipeline kullanıyor — mimari parçalı
- `run_inference_binary.py` — API'dan çağrılmıyor, dead path
- `should_run_localization()` hiç kullanılmıyor
- `airesult_mapper` vs manifest artifact discovery çift yol
- sklearn/xgboost/shap unpinned → `InconsistentVersionWarning` joblib load'da
- SHAP global monkey-patch (`shap_ovr.py`) — import side effect
- Grad-CAM `cleanup()` her path'te çağrılmıyor
- `TreeExplainer` her istekte yeniden oluşturuluyor — CPU yükü
- NORM primary iken sanity MI class_idx=0 kullanıyor
- İki manifest writer (`reporting.py` vs `pipeline.py`)

### Frontend UX / a11y

- `lang="en"` ama UI Türkçe
- Upload zone klavye erişilebilir değil
- Mobile tabs ARIA eksik
- Chat streaming `aria-live` yok
- `TechnicalDetails` sahte model_hash / threshold_hash
- ~30 kullanılmayan shadcn component → bundle şişkinliği
- React Query provider var, hiç kullanılmıyor
- LlmStatusBanner eski model picker metni
- Analyze fail → welcome'a dönüyor, dosya seçimi kayboluyor

### Test boşlukları

- Startup fail-closed test yok
- 413 upload size test yok
- Localization explain + PNG serve test yok
- `ensemble_weight=0.15` default test yok
- `api_mapper.py` unit test yok
- XAI sanity unit test yok
- E2E otomasyon yok (sadece `docs/qa_manual_tr.md`)

### Dokümantasyon (stale)

| Dosya | Sorun |
|-------|--------|
| `CLAUDE.md` | Frontend paths legacy; "kalan sorunlar" eski |
| `README.md` | Ensemble 50/50 yazıyor, kod 15/85 |
| `docs/05_frontend_integration.md` | Eski flat SPA |
| `docs/00_repo_map.md` | Dockerfile/requirements "eksik" diyor |
| `docs/MASTER_SOURCE_OF_TRUTH.md` | consistency_guard UNUSED (yanlış) |
| `docs/agents/domain.md` | `CONTEXT.md`, `docs/adr/` referans — yok |
| `docs/02_api_contracts.md` | utcnow, on_event örnekleri eski |

---

## P3 — Düşük öncelik

- Timestamp format tutarsızlığı (`isoformat` vs `+ "Z"`)
- `traceback.print_exc()` production log
- `uvicorn host 0.0.0.0` local `__main__`
- Debug log localStorage quota sessiz fail
- Theme localStorage gereksiz write
- `example.functions.ts` dead code
- `.idea/` tracked (gitignore'a rağmen)
- `pytest.ini` blanket DeprecationWarning ignore
- Git hook manuel enable (`core.hooksPath`)
- `.env.example` yok
- `docker-compose version: "3.8"` deprecated key
- `academic_assets/`, `docs/evidence/` büyük snapshot'lar

---

## Önerilen sprint planı (agent-friendly)

### Sprint A — Doğruluk (1–2 hafta)
1. Inference normalizasyon stats
2. Ensemble weight tek kaynak (`artifacts/thresholds_superclass.json`)
3. ECG input validation + gerçek `input_meta.shape`
4. Feature schema validate at inference

### Sprint B — Production readiness (1 hafta)
5. `asyncio.to_thread` inference
6. CI workflow (pytest + vitest + tsc)
7. Debug endpoint dev-only gate
8. CORS prod config

### Sprint C — DX & docs (3–5 gün)
9. CLAUDE.md + README + docs/05 güncelle
10. `frontend-legacy` deprecate banner
11. `.env.example`, `.python-version`
12. requirements pin / lockfile

### Sprint D — XAI & test (1 hafta)
13. Kırık eval import fix
14. Sanity tests + train_baseline.npz
15. Frontend contract Zod + vitest genişletme
16. Docker frontend architecture kararı

---

## Agent issue şablonu

Her madde GitHub issue'ya şöyle açılabilir:

```
Title: [P0|P1|P2] Kısa başlık
Labels: tech-debt, ready-for-agent (veya ready-for-human)
Body:
- Konum: path:line
- Risk: ...
- Kabul kriteri: ...
- Test: ...
```

---

## Tarama kaynakları

- Backend scan: `src/backend/`, `src/pipeline/`, `src/contracts/`, `tests/`
- Frontend scan: `frontend/src/`
- ML/XAI scan: `src/xai/`, `src/models/`, `src/data/`, `artifacts/`
- Infra scan: Docker, deps, docs, git artifacts

**Sonraki adım:** Bu envanterden P0/P1 maddelerini GitHub Issues'a bölmek (`/to-issues` skill).
